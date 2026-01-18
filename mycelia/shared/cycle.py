import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import bittensor
import requests
from pydantic import BaseModel

from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import (
    MinerChainCommit,
    ValidatorChainCommit,
    WorkerChainCommit,
    get_chain_commits,
    serve_axon,
    _subtensor_lock,
)
from mycelia.shared.config import MinerConfig, ValidatorConfig, WorkerConfig
from mycelia.shared.helper import h256_int, parse_dynamic_filename
from mycelia.validator.evaluator import MinerEvalJob

configure_logging()
logger = structlog.get_logger(__name__)


class PhaseResponse(BaseModel):
    block: int
    cycle_length: int  # how long is one cycle
    cycle_index: int  # which cycle are we in
    cycle_block_index: int  # how far in block are we into a cycle
    phase_name: str  # what is the name of the current phase
    phase_index: int  # what is the id of the phase
    phase_start_block: int  # the start block of the phase
    phase_end_block: int  # the end block of the phase
    blocks_into_phase: int  # how far in block are we in the current phase
    blocks_remaining_in_phase: int  # how manuy block left in the phase


@dataclass
class PhaseNames:
    distribute: str = "Distribute"  # miner download from validator
    train: str = "Train"  # miner trian
    commit: str = "Commit"  # miner commit hash and  vlaidators commit seed
    submission: str = "Submission"  # submit model
    validate: str = "Validate"  # validator validate
    merge: str = "Merge"  # validator merge


def wait_till(config: MinerConfig, phase_name: PhaseNames, subtensor: bittensor.Subtensor = None, poll_fallback_block: int = 3):
    """
    Wait until a specific phase starts.

    Uses local phase computation to determine when to proceed.

    Args:
        config: Worker configuration
        phase_name: Name of phase to wait for
        subtensor: Bittensor subtensor instance (optional, uses decentralized if provided)
        poll_fallback_block: Minimum poll interval in blocks

    Returns:
        (should_submit, phase_end_block)
    """
    should_submit = False
    logger.info(f"<{phase_name}> waiting to begin...")
    while not should_submit:
        should_submit, blocks_till, phase_response = should_act(config, phase_name, subtensor)
        if should_submit is False and blocks_till > 0:
            sleep_sec = min(blocks_till, max(poll_fallback_block, blocks_till * 0.9)) * 12

            check_time = datetime.now() + timedelta(seconds=sleep_sec)
            check_time_str = check_time.strftime("%H:%M:%S")

            logger.info(f"<{phase_name}> to begin in {blocks_till} blocks, check again at {check_time_str}")
            time.sleep(sleep_sec)

    logger.info(f"<{phase_name}> has started, {phase_response.blocks_remaining_in_phase} blocks left in phase.")
    return should_submit, phase_response.phase_end_block


def should_act(config: MinerConfig, phase_name: PhaseNames, subtensor: bittensor.Subtensor = None) -> tuple[bool, int, int]:
    """
    Check if we should act in the given phase (DECENTRALIZED).

    Args:
        config: Worker configuration
        phase_name: Phase to check for
        subtensor: Bittensor subtensor instance (optional, uses decentralized if provided)

    Returns:
        (should_submit, blocks_till, phase_response)
    """
    phase_response: PhaseResponse = get_phase(config, subtensor)
    should_submit = phase_response.phase_name == phase_name
    blocks_till = get_blocks_until_next_phase(config, subtensor)[phase_name]
    return should_submit, blocks_till, phase_response


def search_model_submission_destination(
    wallet: bittensor.Wallet,
    config: MinerConfig,
    subtensor: bittensor.Subtensor,
    allow_fallback: bool = True,
) -> bittensor.Axon | None:
    """
    Find validator to submit model.

    Submission routing:
    1. Prefer assigned validator (deterministic assignment)
    2. If assigned validator unreachable/inactive, fallback to any validator in group

    Args:
        wallet: Miner's wallet
        config: Miner configuration
        subtensor: Bittensor subtensor instance
        allow_fallback: Allow fallback to any validator if assigned is unreachable

    Returns:
        Axon of validator to submit to, or None if no validator available
    """
    miner_hotkey = wallet.hotkey.ss58_address
    validator_miner_assignment = get_validator_miner_assignment(config, subtensor)

    # Find assigned validator
    assigned_validator_hotkey = None
    for validator_hotkey, assigned_miners in validator_miner_assignment.items():
        if miner_hotkey in assigned_miners:
            assigned_validator_hotkey = validator_hotkey
            break

    if assigned_validator_hotkey is not None:
        try:
            metagraph = subtensor.metagraph(netuid=config.chain.netuid)
            uid = metagraph.hotkeys.index(assigned_validator_hotkey)
            axon = metagraph.axons[uid]

            logger.info(
                "Found assigned validator for submission",
                validator=assigned_validator_hotkey[:16] + "...",
                miner=miner_hotkey[:16] + "...",
            )
            return axon

        except (ValueError, IndexError) as e:
            logger.warning(
                f"Assigned validator not reachable: {e}",
                validator=assigned_validator_hotkey[:16] + "...",
            )
            # Fall through to fallback logic

    # Fallback: submit to any validator in group
    if allow_fallback and validator_miner_assignment:
        # Pick first validator with assignment (early days: submit to all strategy)
        fallback_validator_hotkey = next(iter(validator_miner_assignment.keys()))

        try:
            metagraph = subtensor.metagraph(netuid=config.chain.netuid)
            uid = metagraph.hotkeys.index(fallback_validator_hotkey)
            axon = metagraph.axons[uid]

            logger.info(
                "Using fallback validator for submission",
                validator=fallback_validator_hotkey[:16] + "...",
                miner=miner_hotkey[:16] + "...",
                reason="assigned_validator_unreachable" if assigned_validator_hotkey else "no_assignment",
            )
            return axon

        except (ValueError, IndexError) as e:
            logger.error(f"Fallback validator not reachable: {e}")

    logger.error(
        "No validator available for submission",
        miner=miner_hotkey[:16] + "...",
        has_assignment=assigned_validator_hotkey is not None,
    )
    return None


def setup_chain_worker(config):
    wallet = bittensor.Wallet(name=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)
    subtensor = bittensor.Subtensor(network=config.chain.network)
    serve_axon(
        config=config,
        wallet=wallet,
        subtensor=subtensor,
    )
    return wallet, subtensor


def assign_miners_to_validators_old(
    validators: dict[str, Any],  # {validator_id: seed}
    miners: list[str],
) -> dict[str, list[str]]:
    """
    OLD IMPLEMENTATION - Kept for backward compatibility.
    Use get_validator_miner_assignment() instead .
    """
    n_v = len(validators)
    n_m = len(miners)

    if n_v == 0:
        raise ValueError("No validators provided")

    # --- 0) Combined seed (hash of all validator seeds)
    combined_seed_str = "".join(str(validators[v]) for v in sorted(validators.keys()))
    combined_seed = hashlib.sha256(combined_seed_str.encode()).hexdigest()

    # --- 1) Balanced capacities
    base = n_m // n_v
    rem = n_m % n_v
    v_ids = list(validators.keys())

    ranked_for_bonus = sorted(
        v_ids,
        key=lambda vid: h256_int("cap_bonus", validators[vid], combined_seed),
        reverse=True,
    )
    capacities = {vid: base for vid in v_ids}
    for vid in ranked_for_bonus[:rem]:
        capacities[vid] += 1

    # --- 2) Deterministic miner order seeded by combined validator seed
    miners_sorted = sorted(miners, key=lambda mid: h256_int("miner_order", mid, combined_seed))

    # --- 3) Preference per miner (based on validator seed + combined seed)
    def validator_prefs(mid: str) -> list[str]:
        return sorted(
            v_ids,
            key=lambda vid: h256_int("preference", mid, validators[vid], combined_seed),
            reverse=True,
        )

    # --- 4) Assign miners evenly, respecting capacities
    assignment: dict[str, list[str]] = {vid: [] for vid in v_ids}
    for mid in miners_sorted:
        prefs = validator_prefs(mid)
        for vid in prefs:
            if capacities[vid] > 0:
                assignment[vid].append(mid)
                capacities[vid] -= 1
                break
        else:
            # Should never happen if capacities sum == len(miners)
            assignment[prefs[-1]].append(mid)

    return assignment


def get_combined_validator_seed(config: WorkerConfig, subtensor: bittensor.Subtensor) -> str:
    """
    Deterministically combine validator seeds into a single hex string.

    We sort validator IDs so the result is independent of dict iteration order.
    """
    commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(config, subtensor)

    validator_seeds = get_validator_seed_from_commit(config, commits)
    if not validator_seeds:
        raise ValueError("No validators provided")

    combined_seed_str = "".join(str(validator_seeds[v]) for v in sorted(validator_seeds.keys()))
    return hashlib.sha256(combined_seed_str.encode()).hexdigest()


def get_validator_miner_assignment(config: WorkerConfig, subtensor: bittensor.Subtensor):
    """
    Get deterministic validator↔miner assignment using Isabella's algorithm.

    This implements the decentralized assignment:
    1. Get active validators + their seeds from chain
    2. Get miners in expert group from chain
    3. Compute group seed from validator seeds
    4. Deterministically assign miners to validators

    Args:
        config: Worker configuration
        subtensor: Bittensor subtensor instance

    Returns:
        Dict mapping validator_hotkey → list[miner_hotkeys]
    """
    from mycelia.shared.chain import (
        assign_miners_to_validators,
        compute_group_seed,
        get_active_validators,
        get_cycle_anchor_block,
        get_phase_periods_from_config,
        get_seed_commit_snapshot_block_hash,
        read_all_commitments,
    )

    try:
        current_block = subtensor.block
        anchor_block = get_cycle_anchor_block(current_block, config.cycle.cycle_length)
        anchor_block_hash = subtensor.get_block_hash(anchor_block)

        # Get phase periods from config (deterministic)
        phase_periods = get_phase_periods_from_config(config)

        # Compute seed commit block hash (DECENTRALIZED - from config + anchor)
        # CRITICAL: Use END of commit window so validators have time to commit
        seed_commit_block_hash = get_seed_commit_snapshot_block_hash(
            subtensor=subtensor,
            anchor_block=anchor_block,
            phase_periods=phase_periods,
        )

        # Get active validators with their seeds at seed commit phase
        active_validators = get_active_validators(
            subtensor=subtensor,
            netuid=config.chain.netuid,
            expert_group_id=config.task.expert_group_id,
            seed_commit_block_hash=seed_commit_block_hash,
        )

        if not active_validators:
            logger.warning(
                "No active validators found for assignment",
                expert_group=config.task.expert_group_id,
            )
            return {}

        # Get miners in expert group from commitments
        all_commits = read_all_commitments(subtensor, config.chain.netuid)
        metagraph = subtensor.metagraph(config.chain.netuid)

        miners_in_group = []
        for neuron in metagraph.neurons:
            if neuron.validator_permit:
                continue  # Skip validators

            hotkey = neuron.hotkey
            commit = all_commits.get(hotkey)
            if not commit:
                continue

            # Check expert group
            miner_expert_group = commit.get('e') or commit.get('expert_group')
            if miner_expert_group == config.task.expert_group_id:
                miners_in_group.append(hotkey)

        if not miners_in_group:
            logger.warning(
                "No miners found in expert group",
                expert_group=config.task.expert_group_id,
            )
            return {}

        # Compute group seed (order-independent)
        group_seed = compute_group_seed(
            netuid=config.chain.netuid,
            expert_group_id=config.task.expert_group_id,
            anchor_block_hash=anchor_block_hash,
            validator_seeds=active_validators,
        )

        # Deterministic assignment
        validator_list = list(active_validators.keys())
        assignments = assign_miners_to_validators(
            miners=miners_in_group,
            validators=validator_list,
            group_seed=group_seed,
        )

        logger.info(
            "Computed validator↔miner assignments",
            expert_group=config.task.expert_group_id,
            num_validators=len(validator_list),
            num_miners=len(miners_in_group),
            group_seed=group_seed,
        )

        return assignments

    except Exception as e:
        logger.error(f"Error computing validator↔miner assignment: {e}", exc_info=True)
        # Fallback to old implementation
        commits: tuple[WorkerChainCommit, bittensor.Neuron] = get_chain_commits(config, subtensor)
        validator_seeds = get_validator_seed_from_commit(config, commits)
        miners = get_miners_from_commit(config, commits)
        return assign_miners_to_validators_old(validator_seeds, miners)  # type: ignore


def get_validator_seed_from_commit(config, commits):
    validator_seeds: dict[str, int] = {
        neuron.hotkey: commit.miner_seed
        for commit, neuron in commits
        if isinstance(commit, ValidatorChainCommit)
        and getattr(commit, "expert_group", None) == config.task.expert_group_id
    }
    return validator_seeds


def get_miners_from_commit(config, commits):
    miners: list[str] = [
        neuron.hotkey
        for commit, neuron in commits
        if isinstance(commit, MinerChainCommit) and getattr(commit, "expert_group", None) == config.task.expert_group_id
    ]

    return miners


def get_phase(config: WorkerConfig, subtensor: bittensor.Subtensor) -> PhaseResponse:
    """
    Determine current phase based on block schedule (DECENTRALIZED).

    Uses local computation from cycle anchor + phase schedule.
    No centralized service required - all nodes compute same result.

    Args:
        config: Worker configuration with cycle settings
        subtensor: Bittensor subtensor instance (REQUIRED for decentralized computation)

    Returns:
        PhaseResponse with current phase information

    Raises:
        ValueError if subtensor is None (centralized mode removed)
    """
    if subtensor is None:
        raise ValueError(
            "subtensor is required for decentralized phase computation. "
            "Centralized phase service has been removed. "
            "Always pass subtensor to get_phase()."
        )

    from mycelia.shared.chain import (
        get_cycle_anchor_block,
        get_phase_periods_from_config,
    )

    # Get current block
    current_block = subtensor.block

    # Compute cycle anchor and phase schedule
    cycle_length = config.cycle.cycle_length
    anchor_block = get_cycle_anchor_block(current_block, cycle_length)
    phase_periods = get_phase_periods_from_config(config)

    # Determine which phase we're in
    cycle_index = current_block // cycle_length
    cycle_block_index = current_block % cycle_length

    offset = 0
    for phase_index, period in enumerate(phase_periods):
        phase_start = anchor_block + offset
        phase_end = phase_start + period

        if current_block >= phase_start and current_block < phase_end:
            phase_names = [
                PhaseNames.distribute,
                PhaseNames.train,
                PhaseNames.commit,
                PhaseNames.submission,
                PhaseNames.validate,
                PhaseNames.merge,
            ]

            return PhaseResponse(
                block=current_block,
                cycle_length=cycle_length,
                cycle_index=cycle_index,
                cycle_block_index=cycle_block_index,
                phase_name=phase_names[phase_index],
                phase_index=phase_index,
                phase_start_block=phase_start,
                phase_end_block=phase_end,
                blocks_into_phase=current_block - phase_start,
                blocks_remaining_in_phase=phase_end - current_block,
            )

        offset += period

    # Should never reach here if phase_periods sum to cycle_length
    logger.error(
        "Current block not in any phase - config error?",
        current_block=current_block,
        anchor_block=anchor_block,
        cycle_length=cycle_length,
        phase_periods=phase_periods,
    )
    raise ValueError(f"Block {current_block} not in any phase")


def get_blocks_until_next_phase(config: WorkerConfig, subtensor: bittensor.Subtensor) -> dict[str, int]:
    """
    Get blocks until each phase starts (DECENTRALIZED).

    Uses local computation from cycle anchor + phase schedule.

    Args:
        config: Worker configuration
        subtensor: Bittensor subtensor instance (REQUIRED)

    Returns:
        Dict mapping phase_name → blocks_until_phase_starts

    Raises:
        ValueError if subtensor is None
    """
    if subtensor is None:
        raise ValueError(
            "subtensor is required for decentralized phase computation. "
            "Centralized phase service has been removed."
        )

    from mycelia.shared.chain import (
        get_cycle_anchor_block,
        get_phase_periods_from_config,
    )

    # Get current block
    current_block = subtensor.block

    # Compute phase schedule
    cycle_length = config.cycle.cycle_length
    anchor_block = get_cycle_anchor_block(current_block, cycle_length)
    phase_periods = get_phase_periods_from_config(config)

    phase_names = [
        PhaseNames.distribute,
        PhaseNames.train,
        PhaseNames.commit,
        PhaseNames.submission,
        PhaseNames.validate,
        PhaseNames.merge,
    ]

    result = {}
    offset = 0
    for phase_index, period in enumerate(phase_periods):
        phase_start = anchor_block + offset
        blocks_until = phase_start - current_block

        if blocks_until < 0:
            # Phase already started - time until next cycle's phase
            blocks_until += cycle_length

        result[phase_names[phase_index]] = blocks_until
        offset += period

    return result


def get_blocks_from_previous_phase(config: WorkerConfig, subtensor: bittensor.Subtensor) -> dict[str, tuple[int, int]]:
    """
    Get block ranges for previous phases in current cycle (DECENTRALIZED).

    Uses local computation from cycle anchor + phase schedule.

    Args:
        config: Worker configuration
        subtensor: Bittensor subtensor instance (REQUIRED)

    Returns:
        Dict mapping phase_name → (start_block, end_block) for previous phases

    Raises:
        ValueError if subtensor is None
    """
    if subtensor is None:
        raise ValueError(
            "subtensor is required for decentralized phase computation. "
            "Centralized phase service has been removed."
        )

    from mycelia.shared.chain import (
        get_cycle_anchor_block,
        get_phase_periods_from_config,
    )

    # Get current block
    current_block = subtensor.block

    # Compute phase schedule
    cycle_length = config.cycle.cycle_length
    anchor_block = get_cycle_anchor_block(current_block, cycle_length)
    phase_periods = get_phase_periods_from_config(config)

    phase_names = [
        PhaseNames.distribute,
        PhaseNames.train,
        PhaseNames.commit,
        PhaseNames.submission,
        PhaseNames.validate,
        PhaseNames.merge,
    ]

    result = {}
    offset = 0
    for phase_index, period in enumerate(phase_periods):
        phase_start = anchor_block + offset
        phase_end = phase_start + period

        # Only include phases that have already ended
        if phase_end <= current_block:
            result[phase_names[phase_index]] = (phase_start, phase_end)

        offset += period

    return result


def load_submission_files(folder: str = "miner_submission"):
    """
    Scans a folder for .pt files and returns:
        { filename: {parsed key/values} }
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

    files_dict = {}
    for file_name in folder_path.glob("*.pt"):
        meta = parse_dynamic_filename(file_name.name)
        files_dict[file_name.name] = meta

    return files_dict


def gather_validation_job(config: ValidatorConfig, subtensor: bittensor.Subtensor, step: int) -> list[MinerEvalJob]:
    """
    Gather validation jobs for miners (DECENTRALIZED).

    Uses local phase computation to determine submission window.

    Args:
        config: Validator configuration
        subtensor: Bittensor subtensor instance
        step: Current global optimization step

    Returns:
        List of miner evaluation jobs for this validator
    """
    validator_miner_assignment = get_validator_miner_assignment(config, subtensor)
    miner_assignment = validator_miner_assignment.get(config.chain.hotkey_ss58, [])
    miner_submission_files = load_submission_files(str(config.ckpt.miner_submission_path))

    # Use decentralized phase computation (pass subtensor)
    previous_phases = get_blocks_from_previous_phase(config, subtensor)

    if PhaseNames.submission not in previous_phases:
        logger.warning("Submission phase not yet completed in this cycle")
        return []

    previous_phase_range = previous_phases[PhaseNames.submission]

    hotkeys = subtensor.metagraph(netuid=config.chain.netuid).hotkeys
    miner_jobs = []
    for file_name, submission_meta in miner_submission_files.items():
        if (
            submission_meta["hotkey"] in miner_assignment
            and submission_meta["block"] >= previous_phase_range[0]
            and submission_meta["block"] <= previous_phase_range[1]
        ):
            miner_jobs.append(
                MinerEvalJob(
                    uid=hotkeys.index(submission_meta["hotkey"]),
                    hotkey=submission_meta["hotkey"],
                    model_path=config.ckpt.miner_submission_path / file_name,
                    step=step,
                )
            )

    return miner_jobs
