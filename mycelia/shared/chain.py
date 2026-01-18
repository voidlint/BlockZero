from __future__ import annotations

import base64
import hashlib
import json
import random
import threading
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import bittensor
import torch
from pydantic import BaseModel, ConfigDict, Field

from mycelia.shared.app_logging import structlog
from mycelia.shared.checkpoint import ModelMeta, delete_old_checkpoints
from mycelia.shared.client import download_model
from mycelia.shared.config import WorkerConfig
from mycelia.shared.schema import verify_message

logger = structlog.get_logger(__name__)

# Global lock for subtensor WebSocket access to prevent concurrent recv calls
_subtensor_lock = threading.Lock()

# Phase indices (must match cycle phase ordering)
# Phases: [distribute, train, commit, submission, validate, merge]
PHASE_INDEX_DISTRIBUTE = 0
PHASE_INDEX_TRAIN = 1
PHASE_INDEX_COMMIT = 2  # Seed commit phase (validators commit seeds + signed model hash)
PHASE_INDEX_SUBMISSION = 3
PHASE_INDEX_VALIDATE = 4
PHASE_INDEX_MERGE = 5


# --- Status structure and submission (for miner validator communication)---
class WorkerChainCommit(BaseModel):
    ip: str
    port: int
    active: bool
    stake: float
    validator_permit: bool


class ValidatorChainCommit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    model_hash: str | None = Field(default=None, alias="h")
    signed_model_hash: str | None = Field(default=None, alias="sh")  # For 2-phase commit
    global_ver: int | None = Field(default=None, alias="v")
    expert_group: int | None = Field(default=None, alias="e")
    miner_seed: int | None = Field(default=None, alias="s")
    block: int | None = Field(default=None, alias="b")
    commit_phase: int | None = Field(default=None, alias="p")  # 1 or 2 for 2-phase commit


class MinerChainCommit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    block: int = Field(alias="b")
    expert_group: int | None = Field(default=None, alias="e")
    model_hash: str | None = Field(default=None, alias="h")
    global_ver: int | None = Field(default=0, alias="v")
    inner_opt: int | None = Field(default=0, alias="i")


def commit_status(
    config: WorkerConfig,
    wallet: bittensor.Wallet,
    subtensor: bittensor.Subtensor,
    status: ValidatorChainCommit | MinerChainCommit,
) -> None:
    """
    Commit the worker status to chain.

    If encrypted=False:
        - Uses subtensor.set_commitment (plain metadata, immediately visible).

    If encrypted=True:
        - Timelock-encrypts the status JSON using Drand.
        - Stores it via the Commitments pallet so it will be revealed later
          when the target Drand round is reached.

    Assumes:
        - config.chain.netuid: subnet netuid
        - config.chain.timelock_rounds_ahead: how many Drand rounds in the future
          you want the data to be revealed (fallback to 200 if missing).
    """
    # Serialize status first; same input for both plain + encrypted paths
    data_dict = status.model_dump(by_alias=True)

    data = json.dumps(data_dict)

    subtensor.set_commitment(wallet=wallet, netuid=config.chain.netuid, data=data, raise_error=True)

    logger.info("Committed status to chain", status=data_dict)
    return data_dict


def get_chain_commits(
    config: WorkerConfig, subtensor: bittensor.Subtensor, wait_to_decrypt: bool = False
) -> tuple[WorkerChainCommit, bittensor.Neuron]:
    """
    Get all chain commits and classify them as ValidatorChainCommit or MinerChainCommit.

    CRITICAL: Classification uses neuron.validator_permit, NOT presence of miner_seed field.
    This prevents phase-2 validator commits (without seed) from being misclassified as miner commits.

    Args:
        config: Worker configuration
        subtensor: Bittensor subtensor instance
        wait_to_decrypt: Unused (legacy parameter)

    Returns:
        List of (commit, neuron) tuples
    """
    all_commitments = subtensor.get_all_commitments(netuid=config.chain.netuid)
    metagraph = subtensor.metagraph(netuid=config.chain.netuid)

    parsed = []

    for hotkey, commit in all_commitments.items():
        try:
            uid = metagraph.hotkeys.index(hotkey)
            neuron = metagraph.neurons[uid]
        except (ValueError, IndexError):
            logger.warning(f"Hotkey {hotkey[:16]}... not found in metagraph")
            continue

        try:
            status_dict = json.loads(commit)

            # CRITICAL: Use neuron.validator_permit to classify, not field presence
            # Phase-2 validator commits may not have miner_seed, but are still validator commits
            if neuron.validator_permit:
                chain_commit = ValidatorChainCommit.model_validate(status_dict)
            else:
                chain_commit = MinerChainCommit.model_validate(status_dict)

        except Exception as e:
            logger.debug(f"Failed to parse commit for {hotkey[:16]}...: {e}")
            chain_commit = None

        parsed.append((chain_commit, neuron))

    return parsed


# --- setup chain worker ---
def setup_chain_worker(config):
    wallet = bittensor.Wallet(name=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)
    subtensor = bittensor.Subtensor(network=config.chain.network)
    serve_axon(
        config=config,
        wallet=wallet,
        subtensor=subtensor,
    )
    return wallet, subtensor


def serve_axon(config: WorkerConfig, wallet: bittensor.Wallet, subtensor: bittensor.Subtensor):
    try:
        axon = bittensor.Axon(wallet=wallet, external_port=config.chain.port, ip=config.chain.ip)
        axon.serve(netuid=config.chain.netuid, subtensor=subtensor)
    except Exception as e:
        # Non-fatal: allows local testing without chain registration
        logger.warning(f"Failed to serve axon (continuing anyway): {e}")


# --- Chain weight submission ---
def submit_weights_to_chain(
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    netuid: int,
    score_aggregator,  # MinerScoreAggregator
    min_score_threshold: float = 0.0,
) -> bool:
    """
    Submit validator weights to chain based on miner scores.
    
    Args:
        subtensor: Bittensor subtensor instance
        wallet: Validator's wallet
        netuid: Network UID
        score_aggregator: MinerScoreAggregator with miner scores
        min_score_threshold: Minimum score to include miner (default: 0.0)
        
    Returns:
        True if weights were successfully set
    """
    try:
        # Get metagraph to map hotkeys to UIDs
        metagraph = subtensor.metagraph(netuid=netuid)
        
        # Get all miner scores
        uid_score_pairs = score_aggregator.uid_score_pairs()
        
        if not uid_score_pairs:
            logger.warning("No miner scores to submit weights for")
            return False
        
        # Build UID -> weight mapping
        uids = []
        weights = []
        hotkey_to_uid = {n.hotkey: n.uid for n in metagraph.neurons}
        
        for hotkey, score in uid_score_pairs:
            # Skip miners below threshold
            if score < min_score_threshold:
                continue
            
            # Get UID from hotkey
            if hotkey not in hotkey_to_uid:
                logger.warning(f"Miner hotkey not in metagraph: {hotkey[:16]}...")
                continue
            
            uid = hotkey_to_uid[hotkey]
            uids.append(uid)
            weights.append(score)
        
        if not uids:
            logger.warning("No valid miners to set weights for after filtering")
            return False
        
        # Convert to tensors and normalize
        import torch
        uids_tensor = torch.tensor(uids, dtype=torch.int64)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        
        # Normalize weights to sum to 1
        weights_tensor = weights_tensor / weights_tensor.sum()
        
        # Submit to chain
        logger.info(
            "Submitting weights to chain",
            num_miners=len(uids),
            total_weight=float(weights_tensor.sum()),
        )
        
        success, message = subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids_tensor,
            weights=weights_tensor,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            version_key=0,
        )
        
        if success:
            logger.info(
                "Successfully set weights on chain",
                num_miners=len(uids),
                message=message,
            )
        else:
            logger.error(
                "Failed to set weights on chain",
                message=message,
            )
        
        return success
        
    except Exception as e:
        logger.error(f"Error submitting weights to chain: {e}", exc_info=True)
        return False


# --- Commitment reading from chain ---
def read_commitment(
    subtensor: bittensor.Subtensor,
    hotkey: str,
    netuid: int,
) -> dict | None:
    """
    Read commitment from chain for a specific hotkey.

    Args:
        subtensor: Bittensor subtensor instance
        hotkey: Hotkey to read commitment for
        netuid: Network UID

    Returns:
        Parsed commitment dict or None if not found/invalid
    """
    try:
        all_commitments = subtensor.get_all_commitments(netuid=netuid)

        if hotkey not in all_commitments:
            return None

        commit_str = all_commitments[hotkey]
        commit_dict = json.loads(commit_str)

        return commit_dict

    except Exception as e:
        logger.error(f"Failed to read commitment for {hotkey[:16]}...: {e}")
        return None


def read_commitment_at_block(
    subtensor: bittensor.Subtensor,
    hotkey: str,
    netuid: int,
    block_hash: str,
) -> dict | None:
    """
    Read commitment from chain at a specific block.

    This is CRITICAL for real 2-phase commit verification:
    - Phase 1 commitment at anchor_block_hash
    - Phase 2 commitment at later_block_hash

    Args:
        subtensor: Bittensor subtensor instance
        hotkey: Hotkey to read commitment for
        netuid: Network UID
        block_hash: Block hash to read state at

    Returns:
        Parsed commitment dict or None if not found/invalid
    """
    try:
        # Read historical state using substrate client
        with _subtensor_lock:
            if not hasattr(subtensor, 'substrate') or subtensor.substrate is None:
                logger.error("Subtensor has no substrate client - cannot do historical reads")
                # Must fail if we can't do historical reads - don't fallback to current state
                return None

            # Query historical state at specific block hash
            # Storage key format: Commitments.CommitmentOf(netuid, hotkey)
            result = subtensor.substrate.query(
                module='Commitments',
                storage_function='CommitmentOf',
                params=[netuid, hotkey],
                block_hash=block_hash,
            )

            if result is None or result.value is None:
                logger.debug(
                    f"No commitment found at block for {hotkey[:16]}...",
                    block_hash=block_hash[:16] + "..." if block_hash else "None",
                )
                return None

            # Parse the commitment data
            commit_str = result.value
            if isinstance(commit_str, bytes):
                commit_str = commit_str.decode('utf-8')

            commit_dict = json.loads(commit_str)

            logger.debug(
                f"Read historical commitment for {hotkey[:16]}...",
                block_hash=block_hash[:16] + "..." if block_hash else "None",
                phase=commit_dict.get('p'),
            )

            return commit_dict

    except AttributeError as e:
        logger.error(
            f"Substrate client does not support historical queries: {e}",
            block_hash=block_hash[:16] + "..." if block_hash else "None",
        )
        # CRITICAL: Return None - don't fallback to current state
        return None

    except Exception as e:
        logger.error(
            f"Failed to read commitment at block for {hotkey[:16]}...: {e}",
            block_hash=block_hash[:16] + "..." if block_hash else "None",
        )
        return None


def read_all_commitments(
    subtensor: bittensor.Subtensor,
    netuid: int,
) -> dict[str, dict]:
    """
    Read all commitments from chain.
    
    Args:
        subtensor: Bittensor subtensor instance
        netuid: Network UID
        
    Returns:
        Dict mapping hotkey → parsed commitment dict
    """
    try:
        all_commitments = subtensor.get_all_commitments(netuid=netuid)
        
        parsed = {}
        for hotkey, commit_str in all_commitments.items():
            try:
                commit_dict = json.loads(commit_str)
                parsed[hotkey] = commit_dict
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in commitment for {hotkey[:16]}...")
                continue
        
        return parsed
        
    except Exception as e:
        logger.error(f"Failed to read all commitments: {e}")
        return {}


# --- Deterministic seed and consensus mechanisms ---
def get_cycle_anchor_block(current_block: int, cycle_length: int = 600) -> int:
    """
    Get the anchor block for the current cycle.
    All validators in the same cycle will get the same anchor block.

    Args:
        current_block: Current block number
        cycle_length: Length of one complete cycle (default: 600 blocks)

    Returns:
        Anchor block number (start of current cycle)
    """
    cycle_number = current_block // cycle_length
    anchor_block = cycle_number * cycle_length
    return anchor_block


def get_phase_periods_from_config(config: WorkerConfig) -> list[int]:
    """
    Extract phase periods from config in canonical order.

    Args:
        config: Worker configuration

    Returns:
        List of phase periods: [distribute, train, commit, submission, validate, merge]
    """
    return [
        config.cycle.distribute_period,
        config.cycle.train_period,
        config.cycle.commit_period,
        config.cycle.submission_period,
        config.cycle.validate_period,
        config.cycle.merge_period,
    ]


def get_phase_start_block(anchor_block: int, phase_periods: list[int], phase_index: int) -> int:
    """
    Get the start block for a specific phase within a cycle.

    DECENTRALIZED: No phase service required - computed from config + chain height.

    Args:
        anchor_block: Start block of the cycle
        phase_periods: List of phase periods in blocks [distribute, train, commit, submit, validate, merge]
        phase_index: Which phase (0=distribute, 1=train, 2=commit, 3=submit, 4=validate, 5=merge)

    Returns:
        Block number where the phase starts

    Example:
        phase_periods = [2, 10, 3, 10, 10, 10]  # From config
        phase_index = 2  # commit phase
        Result: anchor_block + 2 + 10 = anchor_block + 12
    """
    if phase_index < 0 or phase_index >= len(phase_periods):
        raise ValueError(f"Invalid phase_index {phase_index}, must be 0-{len(phase_periods)-1}")

    # Sum all periods before this phase
    offset = sum(phase_periods[:phase_index])
    return anchor_block + offset


def get_phase_block_hash(
    subtensor: bittensor.Subtensor,
    anchor_block: int,
    phase_periods: list[int],
    phase_index: int,
) -> str:
    """
    Get the block hash for a specific phase start block.

    DECENTRALIZED: Deterministically computed from anchor + phase schedule.

    Args:
        subtensor: Bittensor subtensor instance
        anchor_block: Start block of the cycle
        phase_periods: List of phase periods in blocks
        phase_index: Which phase

    Returns:
        Block hash for the phase start block
    """
    phase_block = get_phase_start_block(anchor_block, phase_periods, phase_index)
    return subtensor.get_block_hash(phase_block)


def get_seed_commit_snapshot_block_hash(
    subtensor: bittensor.Subtensor,
    anchor_block: int,
    phase_periods: list[int],
) -> str:
    """
    Get the block hash for seed commit snapshot (END of commit window).

    CRITICAL: This must be at the END of the commit phase, not the start,
    to give validators time to commit their seeds. Otherwise different nodes
    will see different "active validators" → fork.

    Args:
        subtensor: Bittensor subtensor instance
        anchor_block: Start block of the cycle
        phase_periods: List of phase periods in blocks [distribute, train, commit, ...]

    Returns:
        Block hash at the end of commit phase
    """
    # Commit phase start
    commit_start = get_phase_start_block(anchor_block, phase_periods, PHASE_INDEX_COMMIT)

    # Commit phase end (last block of commit window)
    commit_end = commit_start + phase_periods[PHASE_INDEX_COMMIT] - 1

    logger.debug(
        "Seed commit snapshot at end of commit window",
        commit_start=commit_start,
        commit_end=commit_end,
        commit_period=phase_periods[PHASE_INDEX_COMMIT],
    )

    return subtensor.get_block_hash(commit_end)


def compute_group_seed(
    netuid: int,
    expert_group_id: int,
    anchor_block_hash: str,
    validator_seeds: dict[str, int],
) -> int:
    """
    Compute order-independent group seed from all validator seeds.

    This is Isabella's algorithm for deterministic validator↔miner assignment.
    All validators compute the same seed regardless of when they run.

    Args:
        netuid: Network UID
        expert_group_id: Expert group ID
        anchor_block_hash: Block hash for the cycle anchor
        validator_seeds: Dict mapping validator hotkey → seed

    Returns:
        Deterministic integer seed for the group
    """
    # Make it order-independent by sorting hotkeys
    items = [(hk, int(s)) for hk, s in validator_seeds.items()]
    items.sort(key=lambda x: x[0])

    payload = {
        "netuid": netuid,
        "expert_group_id": expert_group_id,
        "anchor_block_hash": anchor_block_hash,
        "validator_seeds": items,
    }
    b = json.dumps(payload, sort_keys=True).encode()
    digest = hashlib.sha256(b).digest()
    return int.from_bytes(digest[:8], "big")


def assign_miners_to_validators(
    miners: list[str],
    validators: list[str],
    group_seed: int,
) -> dict[str, list[str]]:
    """
    Deterministic miner→validator assignment.

    This is Isabella's algorithm: shuffle miners with the group seed,
    then assign round-robin to validators.

    IMPORTANT: Uses Random(seed) not random.seed() to avoid global state pollution.

    Args:
        miners: List of miner hotkeys
        validators: List of validator hotkeys
        group_seed: Deterministic seed for shuffling

    Returns:
        Dict mapping validator_hotkey → list[miner_hotkeys]
    """
    if not validators:
        return {}

    # IMPORTANT: Use Random(seed) not random.seed() to avoid global RNG pollution
    rng = random.Random(group_seed)
    miners_shuffled = miners[:]
    rng.shuffle(miners_shuffled)

    assignments = {vk: [] for vk in validators}
    for i, miner_hk in enumerate(miners_shuffled):
        vk = validators[i % len(validators)]
        assignments[vk].append(miner_hk)

    return assignments


def get_active_validators(
    subtensor: bittensor.Subtensor,
    netuid: int,
    expert_group_id: int,
    anchor_block: int,
    seed_commit_block_hash: str,
    commit_phase: int = 1,
) -> dict[str, int]:
    """
    Get active validators with their committed seeds at seed commit phase.

    Active validator = validator_permit == True AND has valid seed commitment.

    CRITICAL: This must be stable and only depend on chain-observable state at
    the seed commit window, NOT on later phases or network reachability.

    This prevents validators from strategically not committing to shrink the denominator.

    Args:
        subtensor: Bittensor subtensor instance
        netuid: Network UID
        expert_group_id: Expert group to filter for
        anchor_block: Anchor block for the cycle
        seed_commit_block_hash: Block hash at seed commit phase to read state from
        commit_phase: Expected commit phase (should always be 1 for seed commits)

    Returns:
        Dict mapping validator_hotkey → seed (only active validators)
    """
    try:
        # Handle None seed_commit_block_hash
        if seed_commit_block_hash is None:
            logger.warning("seed_commit_block_hash is None, returning empty active validators")
            return {}
        
        metagraph = subtensor.metagraph(netuid)

        active_validators = {}

        for neuron in metagraph.neurons:
            # Must have validator permit
            if not neuron.validator_permit:
                continue

            hotkey = neuron.hotkey

            # Read commitment at seed commit block (STABLE - doesn't change)
            commit = read_commitment_at_block(
                subtensor=subtensor,
                hotkey=hotkey,
                netuid=netuid,
                block_hash=seed_commit_block_hash,
            )

            if not commit:
                logger.debug(f"Validator {hotkey[:16]}... has no seed commitment")
                continue

            # Check expert group matches
            commit_expert_group = commit.get('e') or commit.get('expert_group')
            if commit_expert_group != expert_group_id:
                logger.debug(
                    f"Validator {hotkey[:16]}... is in different expert group: {commit_expert_group}"
                )
                continue

            # Check commit phase matches (should be phase 1 for seed commits)
            validator_commit_phase = commit.get('p') or commit.get('commit_phase')
            if validator_commit_phase != commit_phase:
                logger.debug(
                    f"Validator {hotkey[:16]}... has wrong commit phase: {validator_commit_phase}"
                )
                continue

            # Validator's seed (miner_seed field in ValidatorChainCommit)
            seed = commit.get('s') or commit.get('miner_seed')
            if seed is None:
                logger.debug(f"Validator {hotkey[:16]}... has no seed in commitment")
                continue

            # This validator is active (has seed commitment at the canonical block)
            active_validators[hotkey] = seed
            logger.debug(
                f"Active validator: {hotkey[:16]}... with seed {seed}",
                expert_group=expert_group_id,
            )

        logger.info(
            f"Found {len(active_validators)} active validators at seed commit block",
            expert_group=expert_group_id,
            total_validators=sum(1 for n in metagraph.neurons if n.validator_permit),
            seed_commit_block=seed_commit_block_hash[:16] + "..." if seed_commit_block_hash else "None",
        )

        return active_validators

    except Exception as e:
        logger.error(f"Error getting active validators: {e}", exc_info=True)
        return {}


def get_deterministic_seed(
    subtensor: bittensor.Subtensor, 
    current_block: int,
    netuid: int,
    cycle_length: int = 600,
) -> int:
    """
    Generate deterministic seed from chain state using cycle anchor block.
    All validators in the same cycle get the SAME seed regardless of when they call this.
    
    Args:
        subtensor: Bittensor subtensor instance
        current_block: Current block number
        netuid: Network UID
        cycle_length: Length of one cycle (default: 600 blocks)
        
    Returns:
        Deterministic integer seed
    """
    try:
        # Use cycle anchor block so all validators get same seed
        anchor_block = get_cycle_anchor_block(current_block, cycle_length)
        
        block_hash = subtensor.get_block_hash(anchor_block)
        seed_bytes = hashlib.sha256(f"{block_hash}:{netuid}:{anchor_block}".encode()).digest()
        seed = int.from_bytes(seed_bytes[:8], 'big')
        
        logger.info(
            f"Generated deterministic seed: {seed} for cycle anchor block {anchor_block} "
            f"(current: {current_block})"
        )
        return seed
    except Exception as e:
        logger.error(f"Failed to get deterministic seed, using fallback: {e}")
        # Fallback to anchor block based seed
        anchor_block = get_cycle_anchor_block(current_block, cycle_length)
        return hash(f"{anchor_block}:{netuid}") & 0xFFFFFFFFFFFFFFFF


def get_consensus_miner_list(
    subtensor: bittensor.Subtensor,
    netuid: int,
    expert_group_id: int,
    seed: int,
    max_miners: int = 100,
) -> list[str]:
    """
    Get deterministic list of miners to evaluate.
    All validators must evaluate the same miners for consensus.
    
    REAL IMPLEMENTATION - Filters miners by expert group from chain commitments.
    
    Args:
        subtensor: Bittensor subtensor instance
        netuid: Network UID
        expert_group_id: Expert group to filter for
        seed: Deterministic seed for shuffling
        max_miners: Maximum number of miners to return
        
    Returns:
        List of miner hotkeys to evaluate
    """
    try:
        metagraph = subtensor.metagraph(netuid)
        
        # Get all miners (non-validators)
        all_miners = [n for n in metagraph.neurons if not n.validator_permit]
        
        if not all_miners:
            logger.warning("No miners found in metagraph")
            return []
        
        # Read all commitments from chain
        all_commits = read_all_commitments(subtensor, netuid)
        
        # Filter by expert group using chain commitments
        miners_in_group = []
        for miner in all_miners:
            hotkey = miner.hotkey
            commit = all_commits.get(hotkey)
            
            if not commit:
                # No commitment = miner hasn't announced their expert group
                logger.debug(f"Miner {hotkey[:16]}... has no commitment")
                continue
            
            # Check expert group
            miner_expert_group = commit.get('e') or commit.get('expert_group')
            if miner_expert_group == expert_group_id:
                miners_in_group.append(hotkey)
        
        if not miners_in_group:
            logger.warning(
                f"No miners found in expert group {expert_group_id}",
                total_miners=len(all_miners),
            )
            return []
        
        logger.info(
            f"Found {len(miners_in_group)} miners in expert group {expert_group_id}",
            total_miners=len(all_miners),
        )

        # Deterministically shuffle based on seed
        # IMPORTANT: Use Random(seed) not random.seed() to avoid global RNG pollution
        rng = random.Random(seed)
        rng.shuffle(miners_in_group)

        # Select top N miners
        selected = miners_in_group[:max_miners]
        
        logger.info(
            f"Selected {len(selected)} miners for evaluation",
            expert_group=expert_group_id,
            seed=seed,
        )
        
        return selected
        
    except Exception as e:
        logger.error(f"Error getting consensus miner list: {e}", exc_info=True)
        return []


def verify_model_commit_signature(
    subtensor: bittensor.Subtensor,
    validator_hotkey: str,
    netuid: int,
    expert_group_id: int,
    anchor_block_hash: str,
) -> tuple[bool, str]:
    """
    Verify validator's model commit signature (phase 2).

    This validates that the validator properly signed their model hash
    with the domain-separated message format.

    Args:
        subtensor: Bittensor subtensor instance
        validator_hotkey: Validator's hotkey to verify
        netuid: Network UID
        expert_group_id: Expert group ID
        anchor_block_hash: Anchor block hash for the cycle

    Returns:
        (is_valid, reason_or_model_hash)
        - (True, model_hash) if verification passes
        - (False, reason_string) if verification fails
    """
    try:
        # Read current commitment (should be phase 2 model commit)
        commit = read_commitment(subtensor, validator_hotkey, netuid)

        if not commit:
            logger.debug(f"No commitment found for validator {validator_hotkey[:16]}...")
            return False, "no_commitment"

        # Verify phase 2 (model distribution)
        commit_phase = commit.get('p') or commit.get('commit_phase')
        if commit_phase != 2:
            logger.debug(
                f"Wrong commit phase for model: {commit_phase}",
                validator=validator_hotkey[:16] + "...",
            )
            return False, f"wrong_phase_{commit_phase}"

        # Extract model hash and signature
        model_hash = commit.get('h') or commit.get('model_hash')
        signed_hash = commit.get('sh') or commit.get('signed_model_hash')

        if not model_hash:
            logger.debug(f"Commit missing model_hash", validator=validator_hotkey[:16] + "...")
            return False, "missing_model_hash"

        if not signed_hash:
            logger.debug(f"Commit missing signature", validator=validator_hotkey[:16] + "...")
            return False, "missing_signature"

        # Verify signature over domain-separated message
        # Format: mycelia:v1|netuid|expert_group|anchor_hash|phase|model_hash
        try:
            keypair = bittensor.Keypair(ss58_address=validator_hotkey)

            domain_message = f"mycelia:v1|{netuid}|{expert_group_id}|{anchor_block_hash}|2|{model_hash}"
            message_bytes = domain_message.encode('utf-8')
            signature_bytes = bytes.fromhex(signed_hash)

            is_valid = keypair.verify(message_bytes, signature_bytes)

            if not is_valid:
                logger.warning(
                    "Model signature verification failed",
                    validator=validator_hotkey[:16] + "...",
                    hash=model_hash[:16] + "...",
                )
                return False, "signature_invalid"

            logger.debug(
                "Model signature verified",
                validator=validator_hotkey[:16] + "...",
                hash=model_hash[:16] + "...",
            )
            return True, model_hash

        except Exception as e:
            logger.error(
                f"Signature verification error: {e}",
                validator=validator_hotkey[:16] + "...",
            )
            return False, f"verification_error_{type(e).__name__}"

    except Exception as e:
        logger.error(f"Error verifying model commit: {e}", exc_info=True)
        return False, "verification_error"


def verify_validator_consensus_on_model(
    subtensor: bittensor.Subtensor,
    netuid: int,
    expert_group_id: int,
    block: int,
    cycle_length: int,
    phase_periods: list[int],
    anchor_block: int | None = None,
    min_quorum_ratio: float = 0.6,
    min_consensus_ratio: float = 0.5,
) -> tuple[str | None, list[str]]:
    """
    Check if validators reached consensus on model hash.

    IMPROVED IMPLEMENTATION - Uses active validators to prevent strategic non-commitment.

    Consensus requires:
    1. Quorum: >= min_quorum_ratio of ACTIVE validators have phase2 commits
    2. Majority: >min_consensus_ratio of ACTIVE validators agree on same hash

    Active validator = validator_permit + valid phase-appropriate commitment.
    This prevents validators from strategically not committing to shrink denominator.

    Args:
        subtensor: Bittensor subtensor instance
        netuid: Network UID
        expert_group_id: Expert group to check
        block: Block number to check consensus at
        cycle_length: Cycle length in blocks (from config, must match all nodes)
        phase_periods: Phase periods [distribute, train, commit, submission, validate, merge]
        anchor_block: Anchor block for the cycle (optional, computed if not provided)
        min_quorum_ratio: Minimum ratio of active validators that must have commits (default: 0.6)
        min_consensus_ratio: Minimum ratio of active validators that must agree (default: 0.5)

    Returns:
        (consensus_hash, list_of_agreeing_validator_hotkeys)
        Returns (None, []) if no consensus reached
    """
    try:
        # Get active validators (those with validator_permit + valid commitments)
        if anchor_block is None:
            anchor_block = get_cycle_anchor_block(block, cycle_length=cycle_length)

        # Compute seed commit block hash (DECENTRALIZED - from config + anchor)
        # CRITICAL: Use END of commit window, not start, so validators have time to commit
        try:
            seed_commit_block_hash = get_seed_commit_snapshot_block_hash(
                subtensor=subtensor,
                anchor_block=anchor_block,
                phase_periods=phase_periods,
            )
        except Exception as e:
            logger.error(f"Failed to get seed commit block hash: {e}")
            return None, []

        active_validators_dict = get_active_validators(
            subtensor=subtensor,
            netuid=netuid,
            expert_group_id=expert_group_id,
            anchor_block=anchor_block,
            seed_commit_block_hash=seed_commit_block_hash,
        )

        active_validator_count = len(active_validators_dict)

        if active_validator_count == 0:
            logger.warning(
                "No active validators found for consensus",
                expert_group=expert_group_id,
            )
            return None, []

        # Read all commitments from chain
        all_commits = read_all_commitments(subtensor, netuid)

        # Collect valid validator model hashes
        hash_votes = defaultdict(list)
        validators_with_phase2_commits = 0

        # Get anchor block hash for signature verification
        try:
            anchor_block_hash = subtensor.get_block_hash(anchor_block)
        except Exception as e:
            logger.error(f"Failed to get anchor block hash: {e}")
            return None, []

        for validator_hotkey in active_validators_dict.keys():
            commit = all_commits.get(validator_hotkey)
            if not commit:
                logger.debug(f"Active validator {validator_hotkey[:16]}... has no commitment")
                continue

            # Verify model commit signature (phase 2)
            is_valid, result = verify_model_commit_signature(
                subtensor=subtensor,
                validator_hotkey=validator_hotkey,
                netuid=netuid,
                expert_group_id=expert_group_id,
                anchor_block_hash=anchor_block_hash,
            )
            if not is_valid:
                logger.debug(
                    f"Validator {validator_hotkey[:16]}... signature verification failed: {result}"
                )
                continue

            # result is the model_hash if valid
            model_hash = result
            hash_votes[model_hash].append(validator_hotkey)
            validators_with_phase2_commits += 1

        # Check quorum: >= min_quorum_ratio of active validators have phase2 commits
        if validators_with_phase2_commits < active_validator_count * min_quorum_ratio:
            logger.warning(
                "Quorum not reached",
                validators_with_commits=validators_with_phase2_commits,
                active_validators=active_validator_count,
                required_quorum=int(active_validator_count * min_quorum_ratio),
            )
            return None, []

        # Check for consensus: >min_consensus_ratio of ACTIVE validators (not just committers)
        for model_hash, agreeing_validators in hash_votes.items():
            if len(agreeing_validators) > active_validator_count * min_consensus_ratio:
                logger.info(
                    "Consensus reached on model hash",
                    hash=model_hash[:16] if model_hash else "None",
                    agreeing=len(agreeing_validators),
                    active_validators=active_validator_count,
                    consensus_ratio=len(agreeing_validators) / active_validator_count,
                )
                return model_hash, agreeing_validators

        logger.warning(
            "No consensus reached",
            validators_with_commits=validators_with_phase2_commits,
            active_validators=active_validator_count,
            hash_distribution={h[:16]: len(v) for h, v in hash_votes.items()},
        )
        return None, []

    except Exception as e:
        logger.error(f"Error verifying validator consensus: {e}", exc_info=True)
        return None, []


# --- Get model from chain ---
def scan_chain_for_new_model(
    current_model_meta: ModelMeta | None,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
) -> tuple[bool, list[dict]]:
    """
    Returns:
        should_download: True if a newer model (by version) is available and a majority
                         agree on the model_hash among those newer entries.
        download_meta:   list of dicts with fields: uid, ip, port, model_hash, model_version
                         filtered to entries that (a) are newer and (b) match the majority hash.
    """
    commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(config, subtensor)

    max_model_meta = ModelMeta(
        global_ver=max(
            (c.global_ver for c, n in commits if c is not None and getattr(c, "global_ver", None) is not None),
            default=0,
        )
    )

    if current_model_meta is not None:
        logger.info(
            "Scan chain - Max model version on chain",
            max_model_version_on_chain=max_model_meta,
        )
        logger.info(
            "Scan chain - Local model version",
            current_model_version=current_model_meta,
        )
        max_model_meta = max(max_model_meta, current_model_meta)

    logger.info(
        "Scan chain - Max model meta",
        max_model_meta=max_model_meta,
    )
    # 0) Download only from validators (use neuron.validator_permit, not miner_seed presence)
    # CRITICAL: Phase-2 validator commits may not have miner_seed, but still have model_hash
    commits = [(c, n) for c, n in commits if n.validator_permit]
    commits = [(c, n) for c, n in commits if getattr(c, "model_hash", False)]

    # 1) collect candidates that are newer than the current version
    most_updated_commits = []  # type: ignore
    for c, n in commits:
        raw_ver = getattr(c, "global_ver", None)
        global_ver = raw_ver if isinstance(raw_ver, int) else 0
        if ModelMeta(global_ver=global_ver) >= max_model_meta:
            most_updated_commits.append((c, n))

    if len(most_updated_commits) == 0:
        return False, []

    # 2) majority filter by model_hash among the newer candidates
    hash_counts = Counter([c.model_hash for c, n in most_updated_commits if getattr(c, "model_hash", False)])
    majority_hash, _count = hash_counts.most_common(1)[0]

    filtered_commits = [(c, n) for c, n in most_updated_commits if getattr(c, "model_hash", False) == majority_hash]
    if not filtered_commits:
        return False, []

    # 3) prepare download_meta for each entry (uid, ip, port, model_hash, model_version)
    download_meta = []
    for commit, neuron in filtered_commits:
        # Only include entries with reachable metadata
        try:
            download_meta.append(
                {
                    "uid": neuron.uid,
                    "ip": neuron.axon_info.ip,
                    "port": neuron.axon_info.port,
                    "model_hash": commit.model_hash,
                    "global_ver": commit.global_ver,
                    "target_hotkey_ss58": neuron.hotkey,
                }
            )

        except Exception:
            logger.info("Cannot append commit", commit=commit)

    # should_download if there is at least one newer entry agreeing on a majority hash
    should_download = len(download_meta) > 0

    return should_download, download_meta


def fetch_model_from_chain(
    current_model_meta: ModelMeta | None,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    expert_group_ids: list = [],
) -> dict | None:
    should_download, download_metas = scan_chain_for_new_model(current_model_meta, config, subtensor)

    logger.info("Fetching model from chain", should_download=should_download, download_metas=download_metas)

    if should_download and download_metas:
        download_success = False
        retries = 0
        max_retries = 3
        base_delay_s = 5  # backoff base

        while (not download_success) and (retries < max_retries):
            for download_meta in download_metas:
                logger.info(f"Downloading from chain: uid = {download_meta['uid']}", download_meta=download_meta)

                # Resolve URL if not provided; fall back to ip/port + default route
                url = download_meta.get("url")
                if not url:
                    ip = download_meta.get("ip")
                    port = download_meta.get("port")
                    # Best-effort defaults; customize if your API differs
                    protocol = getattr(getattr(config, "miner", object()), "protocol", "http")
                    if ip and port:
                        url = f"{protocol}://{ip}:{port}/get-checkpoint"
                    else:
                        logger.warning("Skipping meta without URL or ip:port: %s", download_meta)
                        continue

                out_folder = Path(config.ckpt.validator_checkpoint_path) / (
                    f"uid_{download_meta.get('uid')}_hotkey_{download_meta.get('target_hotkey_ss58')}_globalver_{download_meta.get('global_ver')}"
                )

                out_folder.mkdir(parents=True, exist_ok=True)

                if len(expert_group_ids) == 0:
                    expert_group_ids = [config.task.expert_group_id, "shared"]

                for expert_group_id in expert_group_ids:
                    out_file = (
                        f"model_expgroup_{expert_group_id}.pt"
                        if isinstance(expert_group_id, int)
                        else "model_shared.pt"
                    )
                    out_path = out_folder / out_file
                    try:
                        download_model(
                            url=url,
                            my_hotkey=wallet.hotkey,  # type: ignore
                            target_hotkey_ss58=download_meta["target_hotkey_ss58"],
                            block=subtensor.block,
                            expert_group_id=expert_group_id,
                            token=getattr(config.cycle, "token", ""),
                            out_dir=out_path,
                        )
                        # If download_model doesn't raise, consider it a success
                        download_success = True
                        current_model_version = download_meta["global_ver"]
                        current_model_hash = download_meta["model_hash"]
                        logger.info(
                            "✅ Downloaded checkpoint",
                            out_path=out_path,
                            current_model_version=current_model_version,
                            current_model_hash=current_model_hash,
                        )

                        delete_old_checkpoints(
                            checkpoint_path=Path(config.ckpt.validator_checkpoint_path),
                            topk=config.ckpt.checkpoint_topk,
                        )

                        return download_meta
                    except Exception as e:
                        logger.warning("Download failed", url, e)
                        traceback.print_exc()

            if not download_success:
                retries += 1
                if retries < max_retries:
                    delay = base_delay_s * (2 ** (retries - 1))
                    logger.info("Retrying", delay=delay, retries=retries + 1, max_retries=max_retries)
                    time.sleep(delay)

        if not download_success:
            logger.error(f"❌ All download attempts failed after {retries} retries.")

            return None
