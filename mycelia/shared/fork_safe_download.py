"""
Fork-safe model download with consensus verification.

This module implements fork-safe model downloads that prevent miners from
downloading different models and causing training to split into incompatible branches.
"""

import hashlib
import json
from pathlib import Path

import bittensor

from mycelia.shared.app_logging import structlog
from mycelia.shared.chain import (
    get_cycle_anchor_block,
    get_phase_periods_from_config,
    verify_validator_consensus_on_model,
)
from mycelia.shared.checkpoint import ModelMeta
from mycelia.shared.config import WorkerConfig

logger = structlog.get_logger(__name__)


LAST_FINALIZED_HASH_FILE = Path(".last_finalized_model_hash")


def persist_last_finalized_hash(model_hash: str):
    """
    Persist the last finalized model hash to disk.

    This allows rollback if consensus is later lost.

    Args:
        model_hash: Model hash to persist
    """
    try:
        with open(LAST_FINALIZED_HASH_FILE, 'w') as f:
            json.dump({"model_hash": model_hash}, f)
        logger.debug(f"Persisted last finalized hash: {model_hash[:16]}...")
    except Exception as e:
        logger.warning(f"Failed to persist last finalized hash: {e}")


def load_last_finalized_hash() -> str | None:
    """
    Load the last finalized model hash from disk.

    Returns:
        Model hash or None if not found
    """
    try:
        if LAST_FINALIZED_HASH_FILE.exists():
            with open(LAST_FINALIZED_HASH_FILE, 'r') as f:
                data = json.load(f)
                return data.get("model_hash")
    except Exception as e:
        logger.warning(f"Failed to load last finalized hash: {e}")

    return None


def fetch_model_with_consensus(
    current_model_meta: ModelMeta | None,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    min_quorum_ratio: float = 0.6,
    min_consensus_ratio: float = 0.5,
) -> tuple[bool, str | None, list[str]]:
    """
    Fetch model hash with consensus verification (fork-safe).

    This function REQUIRES strict consensus before downloading:
    - Quorum: ≥60% of active validators have phase2 commits
    - Majority: >50% of active validators agree on same hash

    If consensus is not reached, stay on last finalized model (safe default).

    Args:
        current_model_meta: Current model metadata
        config: Worker configuration
        subtensor: Bittensor subtensor instance
        wallet: Wallet for authentication
        min_quorum_ratio: Minimum ratio of active validators with commits
        min_consensus_ratio: Minimum ratio of active validators agreeing

    Returns:
        (should_download, consensus_hash, agreeing_validators)
        - should_download: True if consensus reached and should download
        - consensus_hash: Model hash with consensus (or None)
        - agreeing_validators: List of validator hotkeys agreeing on hash
    """
    try:
        current_block = subtensor.block
        phase_periods = get_phase_periods_from_config(config)
        cycle_length = config.cycle.cycle_length

        logger.info(
            "Checking validator consensus on model",
            expert_group=config.task.expert_group_id,
            current_block=current_block,
        )

        # Check consensus using decentralized verification
        consensus_hash, agreeing_validators = verify_validator_consensus_on_model(
            subtensor=subtensor,
            netuid=config.chain.netuid,
            expert_group_id=config.task.expert_group_id,
            block=current_block,
            cycle_length=cycle_length,
            phase_periods=phase_periods,
            min_quorum_ratio=min_quorum_ratio,
            min_consensus_ratio=min_consensus_ratio,
        )

        # If NO consensus: stay on last finalized model
        if consensus_hash is None:
            logger.warning(
                "⚠️  No consensus on model hash - staying on last finalized model",
                expert_group=config.task.expert_group_id,
                quorum_required=f"{min_quorum_ratio * 100}%",
                consensus_required=f"{min_consensus_ratio * 100}%",
            )
            return False, None, []

        # If consensus: check if it's a new model
        last_finalized = load_last_finalized_hash()

        if consensus_hash == last_finalized:
            logger.info(
                "Consensus hash matches last finalized - no download needed",
                consensus_hash=consensus_hash[:16] + "...",
            )
            return False, consensus_hash, agreeing_validators

        logger.info(
            "✅ Consensus reached on new model",
            consensus_hash=consensus_hash[:16] + "...",
            agreeing_validators=len(agreeing_validators),
            last_finalized=last_finalized[:16] + "..." if last_finalized else "None",
        )

        return True, consensus_hash, agreeing_validators

    except Exception as e:
        logger.error(
            "Error checking validator consensus",
            error=str(e),
            exc_info=True,
        )
        # Fail safe: don't download on error
        return False, None, []


def download_from_validator(
    validator_hotkey: str,
    consensus_hash: str,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
) -> Path | None:
    """
    Download model from a specific validator with streaming and progress tracking.

    Args:
        validator_hotkey: Validator to download from
        consensus_hash: Expected model hash (for verification)
        config: Worker configuration
        subtensor: Bittensor subtensor instance
        wallet: Wallet for signing download request

    Returns:
        Path to downloaded model or None if download failed
    """
    try:
        # Get validator axon info
        metagraph = subtensor.metagraph(config.chain.netuid)
        uid = metagraph.hotkeys.index(validator_hotkey)
        axon_info = metagraph.axons[uid]

        # Download model using streaming HTTP
        import requests
        from mycelia.shared.schema import construct_block_message, sign_message

        url = f"http://{axon_info.ip}:{axon_info.port}/get-checkpoint"

        logger.info(
            "Downloading model from validator",
            validator=validator_hotkey[:16] + "...",
            url=url,
        )

        # ✅ FIX: Create signed request payload matching validator endpoint expectations
        current_block = subtensor.block
        block_message = construct_block_message(
            target_hotkey_ss58=validator_hotkey,
            block=current_block,
        )
        signature = sign_message(wallet.hotkey, block_message)

        # Form data payload matching /get-checkpoint endpoint
        payload = {
            "target_hotkey_ss58": validator_hotkey,
            "origin_hotkey_ss58": wallet.hotkey.ss58_address,
            "block": current_block,
            "signature": signature,
            "expert_group_id": config.task.expert_group_id,
        }

        # Stream download with progress tracking
        response = requests.post(url, data=payload, timeout=600, stream=True)
        response.raise_for_status()

        # Get file size from headers (if available)
        file_size = int(response.headers.get('Content-Length', 0))

        # Create temporary download path
        download_path = Path(config.ckpt.checkpoint_path) / f"downloading_{consensus_hash[:16]}.pt.tmp"
        final_path = Path(config.ckpt.checkpoint_path) / f"downloaded_{consensus_hash[:16]}.pt"

        # Ensure directory exists
        download_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream download with hash verification
        hash_obj = hashlib.sha256()
        downloaded_bytes = 0
        chunk_size = 8192  # 8KB chunks

        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    hash_obj.update(chunk)
                    downloaded_bytes += len(chunk)

                    # Log progress every 100MB
                    if downloaded_bytes % (100 * 1024 * 1024) < chunk_size:
                        if file_size > 0:
                            progress = (downloaded_bytes / file_size) * 100
                            logger.info(
                                f"Download progress: {progress:.1f}% "
                                f"({downloaded_bytes / (1024**3):.2f}GB / {file_size / (1024**3):.2f}GB)"
                            )
                        else:
                            logger.info(
                                f"Downloaded {downloaded_bytes / (1024**3):.2f}GB"
                            )

        # Verify hash
        downloaded_hash = hash_obj.hexdigest()
        if downloaded_hash != consensus_hash:
            logger.error(
                "Downloaded model hash mismatch",
                expected=consensus_hash[:16] + "...",
                got=downloaded_hash[:16] + "...",
                downloaded_size=downloaded_bytes,
            )
            download_path.unlink()  # Delete corrupted download
            return None

        # Move to final location (atomic rename)
        download_path.rename(final_path)

        logger.info(
            "✅ Model downloaded and verified",
            path=final_path,
            hash=consensus_hash[:16] + "...",
            size_gb=downloaded_bytes / (1024**3),
        )

        return final_path

    except (ValueError, IndexError) as e:
        logger.error(
            f"Validator {validator_hotkey[:16]}... not found in metagraph: {e}"
        )
        return None

    except requests.exceptions.RequestException as e:
        logger.error(
            f"Network error downloading from validator {validator_hotkey[:16]}...: {e}",
            exc_info=True,
        )
        # Clean up partial download
        download_path = Path(config.ckpt.checkpoint_path) / f"downloading_{consensus_hash[:16]}.pt.tmp"
        if download_path.exists():
            download_path.unlink()
        return None

    except Exception as e:
        logger.error(
            f"Error downloading from validator {validator_hotkey[:16]}...: {e}",
            exc_info=True,
        )
        # Clean up partial download
        download_path = Path(config.ckpt.checkpoint_path) / f"downloading_{consensus_hash[:16]}.pt.tmp"
        if download_path.exists():
            download_path.unlink()
        return None


def fetch_model_from_chain_safe(
    current_model_meta: ModelMeta | None,
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
) -> dict | None:
    """
    Fork-safe model download with consensus verification.

    Only downloads if:
    1. Validators have reached consensus on model hash
    2. Quorum + majority thresholds met
    3. All validators agree on same hash

    Otherwise: stay on last finalized model (safe default)

    Args:
        current_model_meta: Current model metadata
        config: Worker configuration
        subtensor: Bittensor subtensor instance
        wallet: Wallet for authentication

    Returns:
        Download metadata dict or None if no download
    """
    # Check consensus
    should_download, consensus_hash, agreeing_validators = fetch_model_with_consensus(
        current_model_meta=current_model_meta,
        config=config,
        subtensor=subtensor,
        wallet=wallet,
        min_quorum_ratio=0.6,
        min_consensus_ratio=0.5,
    )

    if not should_download or consensus_hash is None:
        logger.info("No download needed - staying on current/last finalized model")
        return None

    # Download from agreeing validators (try multiple if needed)
    for validator_hotkey in agreeing_validators:
        download_path = download_from_validator(
            validator_hotkey=validator_hotkey,
            consensus_hash=consensus_hash,
            config=config,
            subtensor=subtensor,
            wallet=wallet,
        )

        if download_path is not None:
            # Download succeeded - persist as last finalized
            persist_last_finalized_hash(consensus_hash)

            return {
                "model_path": str(download_path),
                "model_hash": consensus_hash,
                "downloaded_from": validator_hotkey,
                "agreeing_validators": len(agreeing_validators),
            }

    # All downloads failed - stay on last finalized
    logger.error(
        "❌ All downloads failed - staying on last finalized model",
        consensus_hash=consensus_hash[:16] + "...",
        tried_validators=len(agreeing_validators),
    )

    return None
