from __future__ import annotations

import copy
import asyncio
from dataclasses import dataclass
from pathlib import Path

import requests
import torch
import torch.nn as nn

from mycelia.shared.app_logging import structlog
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.evaluate import evaluate_model
from mycelia.validator.aggregator import MinerScoreAggregator

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MinerEvalJob:
    uid: int
    hotkey: str
    model_path: str
    step: int


# -------------------------- Pipeline Config -----------------------------------
MAX_CONCURRENT_DOWNLOADS = 4
EVAL_WORKERS = 1
DOWNLOAD_TIMEOUT_SEC = 60
EVAL_MAX_BATCHES = 50
# ------------------------------------------------------------------------------


def verify_miner_expert_assignment(
    miner_hotkey: str,
    expert_group_id: int,
    checkpoint_path: Path,
    config,
) -> tuple[bool, str]:
    """
    Verify that miner only trained experts assigned by SN owner.

    This prevents miners from:
    1. Training unauthorized experts
    2. Training experts from other groups
    3. Self-selecting "easier" experts

    Args:
        miner_hotkey: Miner's SS58 address
        expert_group_id: Expert group ID submitted
        checkpoint_path: Path to miner's checkpoint file
        config: Validator config with owner_url

    Returns:
        (is_valid, reason): True if valid, False with reason if invalid
    """
    try:
        # 1. Get authoritative assignment from SN owner
        owner_url = getattr(config, 'owner_url', 'http://localhost:7000')
        response = requests.get(
            f"{owner_url}/get-expert-assignment",
            params={
                "miner_hotkey": miner_hotkey,
                "expert_group_id": expert_group_id,
            },
            timeout=10,
        )

        if response.status_code == 403:
            return False, f"Miner not authorized for expert group {expert_group_id}"

        if response.status_code != 200:
            logger.error(
                "Could not verify expert assignment (SN owner unreachable)",
                status=response.status_code,
            )
            # Fail closed: reject evaluation if SN owner is unreachable
            return False, "verification_failed_owner_unreachable"

        authorized_assignment = response.json()

        # ✅ SECURITY: Verify signature from SN owner
        if "signature" not in authorized_assignment:
            logger.error("Assignment missing signature from SN owner")
            return False, "missing_signature"

        sn_owner_hotkey = authorized_assignment.get("sn_owner_hotkey")
        if not sn_owner_hotkey:
            logger.error("Assignment missing sn_owner_hotkey")
            return False, "missing_sn_owner_hotkey"

        # Verify signature using signature_utils
        from mycelia.shared.signature_utils import verify_assignment_signature

        is_valid = verify_assignment_signature(authorized_assignment, sn_owner_hotkey)
        if not is_valid:
            logger.error(
                "Invalid signature on expert assignment",
                sn_owner=sn_owner_hotkey[:16] + "...",
                miner=miner_hotkey[:16] + "...",
            )
            return False, "invalid_signature"

        logger.debug(
            "Verified signature on expert assignment",
            sn_owner=sn_owner_hotkey[:16] + "...",
        )

        authorized_layers = {
            int(layer_id): {eid for eid, _ in mappings}
            for layer_id, mappings in authorized_assignment["layer_assignments"].items()
        }

        # 2. Load checkpoint and extract which experts were modified
        # ✅ SECURITY FIX: Use safetensors instead of torch.load() to prevent RCE
        try:
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        except Exception as e:
            # Fallback: If not safetensors, use torch.load with weights_only=True (PyTorch 2.0+)
            logger.warning(
                "Checkpoint is not safetensors format, using torch.load with weights_only=True",
                path=str(checkpoint_path),
            )
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
            except Exception as load_err:
                logger.error(
                    "Failed to load checkpoint securely",
                    error=str(load_err),
                    path=str(checkpoint_path),
                )
                return False, f"checkpoint_load_failed_{type(load_err).__name__}"
        
        # Extract expert IDs that have non-zero weights (were trained)
        trained_experts = {}  # {layer_id: set(expert_ids)}
        
        for key in state_dict.keys():
            # Match patterns like: model.layers.5.mlp.experts.2.gate_proj.weight
            # or: language_model.layers.5.mlp.experts.2.gate_proj.weight (VL model)
            if ".experts." in key:
                parts = key.split(".")
                try:
                    # Find layer index
                    layer_idx_pos = parts.index("layers") + 1
                    layer_id = int(parts[layer_idx_pos])
                    
                    # Find expert index
                    expert_idx_pos = parts.index("experts") + 1
                    expert_id = int(parts[expert_idx_pos])
                    
                    if layer_id not in trained_experts:
                        trained_experts[layer_id] = set()
                    trained_experts[layer_id].add(expert_id)
                except (ValueError, IndexError):
                    continue
        
        # 3. Verify trained experts match authorized experts
        for layer_id, expert_ids in trained_experts.items():
            authorized_ids = authorized_layers.get(layer_id, set())
            
            unauthorized = expert_ids - authorized_ids
            if unauthorized:
                logger.warning(
                    "Miner trained unauthorized experts",
                    miner=miner_hotkey[:16] + "...",
                    layer=layer_id,
                    unauthorized=list(unauthorized),
                    authorized=list(authorized_ids),
                )
                return False, f"layer_{layer_id}_unauthorized_experts_{list(unauthorized)}"
        
        # 4. Check if number of experts trained matches assignment
        # If significantly different, miner may have run ESFT locally
        total_authorized = sum(len(ids) for ids in authorized_layers.values())
        total_trained = sum(len(ids) for ids in trained_experts.values())
        
        # Allow some tolerance for partial training/checkpointing
        if total_trained > total_authorized * 1.1:  # 10% tolerance
            logger.warning(
                "Miner trained more experts than authorized (possible ESFT bypass)",
                miner=miner_hotkey[:16] + "...",
                authorized=total_authorized,
                trained=total_trained,
            )
            return False, f"excessive_experts_trained_{total_trained}_vs_{total_authorized}"
        
        logger.info(
            "✓ Miner expert assignment verified",
            miner=miner_hotkey[:16] + "...",
            group=expert_group_id,
            layers_checked=len(trained_experts),
            experts_authorized=total_authorized,
            experts_trained=total_trained,
        )
        return True, "verified"
        
    except Exception as e:
        logger.error(
            "Error during expert assignment verification",
            error=str(e),
            miner=miner_hotkey[:16] + "...",
        )
        # Fail closed: reject evaluation if verification fails
        return False, f"verification_error_{type(e).__name__}"


def load_model_from_path(path: str, base_model, device: torch.device) -> nn.Module:
    """
    Load miner's model from checkpoint.
    CRITICAL: Must return the model WITH miner's weights loaded, not the base model!

    Validates checkpoint compatibility and warns about mismatches.

    ✅ SECURITY: Uses safetensors or weights_only=True to prevent RCE attacks.
    """
    # ✅ SECURITY FIX: Load weights safely without code execution
    try:
        from safetensors.torch import load_file
        sd = load_file(path)
    except Exception:
        # Fallback: torch.load with weights_only=True (PyTorch 2.0+)
        logger.warning(
            "Checkpoint not in safetensors format, using torch.load with weights_only=True",
            path=path,
        )
        try:
            checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
            sd = checkpoint.get("model_state_dict", checkpoint)
        except Exception as e:
            logger.error(
                "Failed to load miner checkpoint securely",
                error=str(e),
                path=path,
            )
            raise ValueError(f"Cannot load checkpoint securely: {e}")

    # Create a copy and load miner's weights into it
    miner_model = copy.deepcopy(base_model)

    # Load with explicit compatibility checking
    missing_keys, unexpected_keys = miner_model.load_state_dict(sd, strict=False)

    # Warn about compatibility issues
    if missing_keys:
        logger.warning(
            f"Miner checkpoint missing {len(missing_keys)} keys - may cause evaluation issues",
            missing_count=len(missing_keys),
            missing_keys=missing_keys[:5] if len(missing_keys) <= 5 else missing_keys[:5] + ["..."],
        )

    if unexpected_keys:
        logger.warning(
            f"Miner checkpoint has {len(unexpected_keys)} unexpected keys - architecture mismatch?",
            unexpected_count=len(unexpected_keys),
            unexpected_keys=unexpected_keys[:5] if len(unexpected_keys) <= 5 else unexpected_keys[:5] + ["..."],
        )

    # Fail-closed if too many missing keys (likely incompatible)
    missing_threshold = len(sd) * 0.1  # Allow up to 10% missing keys
    if len(missing_keys) > missing_threshold:
        raise ValueError(
            f"Miner checkpoint incompatible: {len(missing_keys)} missing keys "
            f"(>{missing_threshold:.0f} threshold). This checkpoint cannot be evaluated safely."
        )

    logger.info(
        "Miner checkpoint loaded",
        total_keys=len(sd),
        missing=len(missing_keys),
        unexpected=len(unexpected_keys),
    )

    # Return the model WITH miner's weights (not the base model!)
    return miner_model.to(device)


async def evaluator_worker(
    name: str,
    config,
    jobs_q: asyncio.Queue[MinerEvalJob],
    aggregator: MinerScoreAggregator,
    device: torch.device,
    base_model: nn.Module,
    tokenizer,
    combinded_seed: str,
    max_eval_batches: int = EVAL_MAX_BATCHES,
    rank: int | None = None,
):
    import gc

    while True:
        job = await jobs_q.get()
        if job is None:  # type: ignore
            jobs_q.task_done()
            logger.debug(f"{name}: shutdown signal received.")
            break

        try:
            # Clear memory before loading
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"{name}: Evaluating hotkey={job.hotkey}")

            # SECURITY: Verify miner trained only assigned experts
            is_valid, reason = await asyncio.to_thread(
                verify_miner_expert_assignment,
                job.hotkey,
                config.task.expert_group_id,
                Path(job.model_path),
                config,
            )
            
            if not is_valid:
                logger.error(
                    f"{name}: Rejecting submission - {reason}",
                    hotkey=job.hotkey,
                    reason=reason,
                )
                aggregator.update(
                    job.hotkey,
                    score=0.0,
                    full_metrics={"reason": reason, "status": "rejected"},
                )
                jobs_q.task_done()
                continue

            # Load model (potentially blocking) in a thread
            model = await asyncio.to_thread(load_model_from_path, job.model_path, base_model, device)
            eval_dataloader = await asyncio.to_thread(
                get_dataloader, config=config, tokenizer=tokenizer, seed=combinded_seed, rank=0, world_size=10
            )

            with torch.inference_mode():
                metrics = await asyncio.to_thread(
                    evaluate_model, job.step, model, eval_dataloader, device, max_eval_batches, rank
                )

            # choose a primary score (here 'accuracy'); adjust if your evaluate_model returns other keys
            score = float(metrics.get("val_loss", 100))
            aggregator.add_score(job.uid, job.hotkey, score)
            logger.info(f"{name}: uid={job.uid} score={score:.4f}")

            # Explicit cleanup
            del eval_dataloader, model, metrics
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"{name}: OOM for uid={job.uid}")
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.exception(f"{name}: Evaluation failed for uid={job.uid}: {e}")
        finally:
            jobs_q.task_done()


async def run_evaluation(
    config, step, device, miners, score_aggregator, base_model: nn.Module, tokenizer, combinded_seed
):
    # Device & dataloader (MOCK). Replace eval_dataloader with a real one.
    miners_q: asyncio.Queue[MinerEvalJob] = asyncio.Queue()

    # Enqueue miners
    for m in miners:
        await miners_q.put(m)

    # Spin up evaluator workers
    eval_workers = [
        asyncio.create_task(
            evaluator_worker(
                f"evaluator-{i+1}", config, miners_q, score_aggregator, device, base_model, tokenizer, combinded_seed
            )
        )
        for i in range(EVAL_WORKERS)
    ]

    # Wait for all miners to be processed
    await miners_q.join()

    # Signal evaluator workers to stop
    for _ in eval_workers:
        await miners_q.put(None)

    await asyncio.gather(*eval_workers)
