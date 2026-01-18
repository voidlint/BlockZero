import datetime
import gc
import os
import time

import bittensor
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

# Set Hugging Face environment variables for offline mode and timeout
# HF_DATASETS_OFFLINE=1 forces use of local cache (bypasses network)
# HF_HUB_READ_TIMEOUT increases timeout for dataset downloads
if "HF_DATASETS_OFFLINE" not in os.environ:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "0")  # Default to online
if "HF_HUB_READ_TIMEOUT" not in os.environ:
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")  # 60 second timeout

from mycelia.miner.train_helper import free_cuda_models, get_status
from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import setup_chain_worker
from mycelia.shared.checkpoint import (
    ModelMeta,
    delete_old_checkpoints,
    get_resume_info,
    load_checkpoint,
    save_checkpoint,
    start_model_from,
)
from mycelia.shared.config import MinerConfig, parse_args
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.esft import (
    apply_esft,
    apply_esft_partition,
    apply_esft_routing,
    apply_esft_with_overlap,
    get_esft_summary,
    measure_expert_overlap,
    measure_expert_routing,
)
from mycelia.shared.evaluate import evaluate_model
from mycelia.shared.expert_manager import ExpertManager
from mycelia.shared.helper import get_model_hash, get_nested_attr
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.model import freeze_parameters, load_model
from mycelia.shared.modeling.mycelia import get_base_tokenizer

configure_logging()
logger = structlog.get_logger(__name__)

# Set environment flag to prevent ESFT execution by miners
os.environ['MYCELIA_ROLE'] = 'miner'

# Optimize HuggingFace downloads to prevent CPU OOM
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # Disable parallel shard downloads
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # Reduce output overhead
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # Longer timeout for large files
os.environ['HF_HUB_MAX_WORKERS'] = '1'  # Single worker to reduce RAM spikes


# this is for local DP only
def init_process(local_rank: int, config: MinerConfig, world_size: int, fn: callable, backend: str | None = None) -> None:
    """
    Initializes the process for distributed training.

    Args:
        rank (int): The rank of the process.
        world_size (int): The total number of processes.
        fn (callable): The function to run for the process.
        backend (str): The backend to use for distributed training.

    Returns:
        None
    """
    os.environ["MASTER_ADDR"] = config.local_par.ip_address
    os.environ["MASTER_PORT"] = str(config.local_par.port)

    if local_rank == 0:
        print(config)  # pretty JSON

    # Auto-select backend based on availability
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # device_id only valid for CUDA devices with nccl backend
    if torch.cuda.is_available() and backend == "nccl":
        device_id = torch.device(f"cuda:{local_rank}")
    else:
        device_id = None

    # Only init process group if actually distributed
    if world_size > 1:
        dist.init_process_group(
            backend,
            rank=local_rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=3600),
            device_id=device_id,
        )

    fn(local_rank, world_size, config)


def setup_training(
    config,
    rank: int,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
    current_model_meta: ModelMeta,
) -> tuple[
    torch.nn.Module,  # model
    torch.optim.Optimizer,  # inner_optimizer
    torch.amp.GradScaler,  # inner_scaler
    torch.optim.lr_scheduler.LRScheduler,  # scheduler
    "ExpertManager",  # em
    StatefulDataLoader,
    dict,  # current model version
    bool,  # use_bf16
]:
    """
    Build model(s), experts layout, optimizers, scheduler, scaler, and optionally resume from a checkpoint.

    Args:
        config: Training/config object with attributes used here (e.g., lr, outer_lr, warmup_steps, etc.).
        rank (int): Process rank.
        device (str | torch.device): Device for the local model (e.g., "cuda:0").

    Returns:
        model (nn.Module): Local (possibly partial) MoE model placed on `device`.
        global_model (nn.Module): Deep-copied global model on CPU, kept in sync with `model`.
        inner_optimizer (Optimizer): Optimizer for `model`.
        outer_optimizer (Optimizer): Optimizer for `global_model`.
        scaler (torch.cuda.amp.GradScaler): GradScaler (enabled iff `config.model.precision == "fp16-mixed"`).
        scheduler (LRScheduler): LR scheduler attached to `inner_optimizer`.
        start_step (int): Step to resume from (0 if starting fresh).
        expert_groups (Sequence[Sequence[int]]): Grouping returned by `create_expert_groups`; typically a list
            (or other sequence) of groups where each group lists the ranks/experts belonging to it.
        group_ids (int): This rank’s group id from `create_expert_groups`.
        expert_manager (ExpertManager): The instantiated ExpertManager for this model/rank.

    Notes:
        - Param group layouts are taken from the *target* optimizers created here.
        - If `resume_from_ckpt` is set and a checkpoint is found, model/opt/scheduler/scaler states are restored
          before syncing `global_model` from `model`.
    """
    logger.info("(0) Setup training")

    # === model & Experts manager ===
    logger.info(f"init - model and expert manager")
    expert_manager = ExpertManager(config)
    # CRITICAL: Load model and replace experts FIRST, then create optimizer
    # This ensures optimizer tracks the new FP32 expert weights, not old 4-bit ones
    model, model_meta = load_model(rank, config, expert_manager, subtensor, wallet)
    model = model.to(device)

    # === Apply ESFT or default freezing ===
    # Qwen3-MoE has 128 experts with 8 active per token, NO shared experts
    # This means only 6.25% of experts get gradients - causing tiny gradients
    # ESFT concentrates training on a subset for 4x stronger gradients
    esft_enabled = get_nested_attr(config, "moe.esft.enabled", False)

    if esft_enabled:
        # Get ESFT configuration
        use_routing_selection = get_nested_attr(config, "moe.esft.use_routing_selection", True)
        total_experts = get_nested_attr(config, "moe.esft.total_experts", 128)
        experts_per_partition = get_nested_attr(config, "moe.esft.experts_per_partition", 32)
        partition_id = get_nested_attr(config, "moe.esft.partition_id", 0)
        top_k = get_nested_attr(config, "moe.esft.top_k_experts_per_layer", 32)

        if use_routing_selection:
            # Method 2: Route-based selection (recommended for single miner)
            logger.info("ESFT enabled - using route-based expert selection")

            # Create dataloader for routing measurement
            routing_dataloader = get_dataloader(
                config, rank=rank, world_size=config.task.data.world_size, tokenizer=tokenizer
            )

            # Measure which experts actually route to our training data
            num_batches = get_nested_attr(config, "moe.esft.measure_routing_batches", 100)
            expert_routing = measure_expert_routing(
                model, routing_dataloader, num_batches=num_batches, device=device
            )

            if expert_routing:
                # Apply route-based ESFT: train top-K routed experts
                # For 128 experts, top_k=32 gives 8/32 = 25% gradient coverage (4x improvement!)
                trainable_experts = apply_esft_routing(model, expert_routing, top_k=top_k)

                # Log ESFT summary
                esft_summary = get_esft_summary(model)
                logger.info("ESFT summary", trainable=esft_summary["trainable"], frozen=esft_summary["frozen"])
            else:
                # Fallback to partition-based if routing measurement failed
                logger.warning("Routing measurement failed - using partition-based ESFT")
                trainable_experts = apply_esft_partition(
                    model,
                    partition_id=partition_id,
                    total_experts=total_experts,
                    experts_per_partition=experts_per_partition,
                )
        else:
            # Method 1: Partition-based (recommended for distributed training)
            logger.info(
                f"ESFT enabled - using partition-based selection (partition {partition_id})"
            )
            trainable_experts = apply_esft_partition(
                model,
                partition_id=partition_id,
                total_experts=total_experts,
                experts_per_partition=experts_per_partition,
            )

            # Log ESFT summary
            esft_summary = get_esft_summary(model)
            logger.info("ESFT summary", trainable=esft_summary["trainable"], frozen=esft_summary["frozen"])
    else:
        # Use original expert group-based freezing
        logger.info("Using default expert group freezing (ESFT disabled)")
        model = freeze_parameters(
            model=model, expert_manager=expert_manager, expert_group_id=config.task.expert_group_id
        )

    # === optimizers ===
    # MUST be created AFTER expert replacement to track correct parameters
    logger.info(f"init - optimizer (after expert replacement)")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    inner_optimizer = torch.optim.AdamW(trainable_params, lr=config.opt.lr, weight_decay=0.1, betas=(0.9, 0.95))

    # Verify optimizer is tracking the correct parameters
    num_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_percent = 100 * (total_params - num_trainable) / total_params if total_params > 0 else 0
    logger.info(
        f"Parameter counts",
        trainable=num_trainable,
        total=total_params,
        frozen_percent=f"{frozen_percent:.1f}%",
        learning_rate=config.opt.lr,
    )

    # Warn if too many parameters are frozen
    if frozen_percent > 95:
        logger.warning(
            "Over 95% of parameters are frozen - gradients may be very small. "
            "Consider unfreezing some attention/mlp layers for better gradient flow."
        )

    # === scheduler === (for inner optimizer)
    logger.info(f"init - scheduler")
    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=config.sched.warmup_steps,
        num_training_steps=config.sched.total_steps,
    )

    # === scaler ===
    logger.info(f"init - inner scaler")
    # Determine precision mode from config
    precision = get_nested_attr(config, "model.precision", "fp16-mixed")
    
    # Check GPU support for bfloat16 (Ampere+ GPUs: 30xx, 40xx, A-series)
    use_bf16 = False
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability(0)
        # Ampere (8.0) or newer supports bfloat16
        use_bf16 = device_capability[0] >= 8
    
    # Use bfloat16 if GPU supports it (better for Qwen3-VL which was pre-trained in bf16)
    # Otherwise fall back to fp16 with scaler
    use_fp16_amp = precision == "fp16-mixed" and torch.cuda.is_available() and not use_bf16
    
    # GradScaler only enabled for FP16 AMP training (NOT for bfloat16)
    inner_scaler = torch.amp.GradScaler(
        "cuda" if torch.cuda.is_available() else "cpu",
        enabled=use_fp16_amp,
    )
    logger.info(f"Precision: {'bfloat16' if use_bf16 else 'float16'}, GradScaler enabled: {use_fp16_amp}")

    # === dataloader ===
    logger.info(f"init - train dataloader")
    train_dataloader = get_dataloader(config, rank=rank, world_size=config.task.data.world_size, tokenizer=tokenizer)

    # === load checkpoint (if any) ===
    logger.info(f"init - load checkpoint")
    resume = False
    latest_checkpoint_path = None
    checkpoint_meta = None
    if get_nested_attr(config, "ckpt.resume_from_ckpt", False):
        resume, checkpoint_meta, latest_checkpoint_path = get_resume_info(rank, config)

    # Only load checkpoint if:
    # 1. Resume is enabled and checkpoint exists
    # 2. Either we're starting fresh (current_model_meta is None/default) OR the checkpoint is newer than current_model_meta
    # This prevents overwriting progress when reloading during training
    is_fresh_start = (
        current_model_meta is None
        or (current_model_meta.inner_opt == 0 and current_model_meta.global_ver == 0)
    )
    should_load_checkpoint = (
        get_nested_attr(config, "ckpt.resume_from_ckpt", False)
        and resume
        and latest_checkpoint_path
        and checkpoint_meta is not None
        and (is_fresh_start or checkpoint_meta > current_model_meta)
    )

    if should_load_checkpoint:
        logger.info(
            "Loading checkpoint",
            checkpoint_path=latest_checkpoint_path,
            checkpoint_meta=checkpoint_meta,
            current_model_meta=model_meta,
        )
        _ = load_checkpoint(
            config=config,
            checkpoint_path=latest_checkpoint_path,
            inner_optimizer=inner_optimizer,
            scheduler=scheduler,
            inner_scaler=inner_scaler,
            rank=rank,
            device=device,
            data_loader=train_dataloader,
        )
        # CRITICAL: Update model_meta with checkpoint metadata so training resumes from correct step
        # Only update if checkpoint is actually newer or we're starting fresh
        if checkpoint_meta is not None and (is_fresh_start or checkpoint_meta > current_model_meta):
            model_meta.inner_opt = checkpoint_meta.inner_opt
            model_meta.global_ver = checkpoint_meta.global_ver
            logger.info(
                "Updated model_meta from checkpoint",
                updated_model_meta=model_meta,
            )
    elif checkpoint_meta is not None and current_model_meta is not None and not is_fresh_start:
        logger.info(
            "Skipping checkpoint load - current model is newer or equal",
            checkpoint_meta=checkpoint_meta,
            current_model_meta=current_model_meta,
        )

    logger.info(f"setup_training: success!")
    return (
        model,
        inner_optimizer,
        inner_scaler,
        scheduler,
        expert_manager,
        train_dataloader,
        model_meta,
        use_bf16,  # Return bfloat16 flag for train_worker
    )


def sum_model_gradients(model):
    """
    Returns the sum of absolute gradients of all model parameters.
    Assumes backward() has already been called.
    """
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total += param.grad.abs().sum().item()
        return total


def train_worker(rank: int, world_size: int, config: MinerConfig) -> None:
    """
    The worker function for training in a distributed setting.

    Args:
        rank (int): The rank of the process.
        world_size (int): The total number of processes.
        config (Config): The configuration object for the training.

    Returns:
        None
    """
    eval_rref = None
    if rank == 0:
        config.write()

    # === set logging ===
    metric_logger = MetricLogger(config, rank)

    # === set up chain worker ===
    wallet, subtensor = setup_chain_worker(config)

    # === mis ===
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    tokenizer = get_base_tokenizer(config)
    
    # Precision mode determined in setup_training and returned
    # Will be set from setup_training return value

    eval_dataloader = get_dataloader(
        config,
        rank=config.local_par.world_size,
        world_size=config.local_par.world_size + 1,
        tokenizer=tokenizer,
    )

    # === set up training ===
    (
        model,
        inner_optimizer,
        inner_scaler,
        scheduler,
        expert_manager,
        train_dataloader,
        current_model_meta,
        use_bf16,  # Get bfloat16 flag from setup_training
    ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta=None)

    # Derive use_fp16_amp from use_bf16 (scaler is already configured in setup_training)
    precision = get_nested_attr(config, "model.precision", "fp16-mixed")
    use_fp16_amp = precision == "fp16-mixed" and torch.cuda.is_available() and not use_bf16

    # === training ===
    loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    training_time = 0
    total_training_time = 0
    training_start_time = None

    inner_optimizer.zero_grad()
    try:
        for step, batch in enumerate(
            iterable=train_dataloader,
            start=max(0, current_model_meta.inner_opt) * config.local_par.gradient_accumulation_steps,
        ):
            # for each step, we run 1 backward
            # for each inner_opt_step, we run local optimization; gradient_accumulation_steps = 1 real step
            # for each global_opt_interval number of inner_opt_step, we synchronise weight from different ddp worker, and then run global optimization

            inner_opt_step = step // config.local_par.gradient_accumulation_steps
            is_inner_optimizer_step = step % config.local_par.gradient_accumulation_steps == 0
            is_start_step = step == current_model_meta.inner_opt * config.local_par.gradient_accumulation_steps
            current_model_meta.inner_opt = inner_opt_step

            # === Training and inner optimization ===
            if is_inner_optimizer_step:
                logger.info(
                    "(1) Start epoch training",
                    step=step,
                    inner_opt_step=inner_opt_step,
                    is_inner_optimizer_step=is_inner_optimizer_step,
                    gradient_accumulation_steps=config.local_par.gradient_accumulation_steps,
                    current_model_meta=current_model_meta,
                )
            if (
                not is_start_step
            ):  # skip training when it is the start step, so that we can benchamrk the original model first
                model.train()
                if training_start_time is None:
                    training_start_time = time.time()

                batch_device = {}
                for key in batch.keys():
                    batch_device[key] = batch[key].to(device)

                # Use appropriate autocast device type and dtype
                # Prefer bfloat16 if GPU supports it (better for Qwen3-VL, no scaler needed)
                autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
                autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16_amp else torch.float32)
                
                with torch.amp.autocast(autocast_device, dtype=autocast_dtype, enabled=torch.cuda.is_available()):
                    outputs = model(**batch_device)
                    loss = outputs.loss / config.local_par.gradient_accumulation_steps
                    # Use aux_loss if available (helps MoE router training)
                    if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                        aux_loss = outputs.aux_loss / config.local_par.gradient_accumulation_steps
                    else:
                        aux_loss = torch.tensor(0, device=device)

                # Skip NaN losses (MPS numerical instability)
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("Skipping batch due to NaN/Inf loss", loss=loss.item() if not torch.isnan(loss) else "NaN")
                    del loss, aux_loss, batch_device, outputs
                    continue
                    
                loss_batch += loss.item()
                aux_loss_batch += aux_loss.item()
                logger.info("training", loss=loss, grad_sum=sum_model_gradients(model))

                # Standard backward pass (no scaling needed for bfloat16)
                # For FP16, scaler handles scaling; for BF16, direct backward is fine
                if inner_scaler.is_enabled():
                    inner_scaler.scale(loss).backward()
                else:
                    loss.backward()

                # === Gradient Health Debug Logging ===
                # Log every 10 optimizer steps to monitor gradient health
                if is_inner_optimizer_step and rank == 0 and inner_opt_step % 10 == 0:
                    non_zero_params = 0
                    total_trainable = 0
                    grad_magnitudes = []

                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            total_trainable += 1
                            if param.grad is not None and param.grad.abs().sum() > 1e-10:
                                non_zero_params += 1
                                grad_magnitudes.append((name.split('.')[-1], float(param.grad.norm().item())))

                    # Sort by magnitude and get top 5
                    top_grads = sorted(grad_magnitudes, key=lambda x: x[1], reverse=True)[:5]

                    logger.info(
                        "=== GRADIENT DEBUG ===",
                        non_zero_gradients=f"{non_zero_params}/{total_trainable}",
                        top_gradient_norms=top_grads,
                        total_grad_sum=f"{sum_model_gradients(model):.2e}",
                        inner_opt_step=inner_opt_step,
                    )

                # === Aggressively free intermediate tensors ===
                del loss, aux_loss, batch_device, outputs
                gc.collect()

            # === inner optimizer ===
            # CRITICAL: Only run optimizer step at accumulation boundaries
            if not is_start_step and is_inner_optimizer_step:
                # Only compute hash on rank 0 and only occasionally to save memory
                old_model_hash = None
                if rank == 0 and step % 100 == 0:  # Only every 100 steps
                    old_model_hash = get_model_hash(model.state_dict())

                # Only do distributed gradient sync if world_size > 1
                if world_size > 1:
                    for n, p in model.named_parameters():
                        if p.grad is None or torch.isnan(p.grad.sum()):
                            continue
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(world_size)

                # Gradient clipping (no unscale_() needed for bfloat16)
                if inner_scaler.is_enabled():
                    # FP16 path: unscale before clipping
                    inner_scaler.unscale_(optimizer=inner_optimizer)
                
                # Collect parameters with valid gradients
                valid_params = [p for p in model.parameters() if p.grad is not None and not torch.isnan(p.grad.sum())]
                if len(valid_params) == 0:
                    logger.warning("No valid gradients found - all gradients are None or NaN")
                    # Skip optimizer step if no valid gradients
                    inner_optimizer.zero_grad()
                    continue
                
                grad_norm = clip_grad_norm_(valid_params, 1.0)

                # Optimizer step
                if inner_scaler.is_enabled():
                    # FP16 path: use scaler
                    scale_before = inner_scaler.get_scale()
                    step_result = inner_scaler.step(inner_optimizer)
                    step_skipped = step_result is None

                    if step_skipped:
                        logger.warning(
                            "GradScaler skipped optimizer step due to inf/NaN gradients",
                            grad_norm=float(grad_norm),
                            scale_before=scale_before,
                        )
                    else:
                        logger.info(
                            "Optimizer step applied (FP16)",
                            grad_norm=float(grad_norm),
                            scale_before=scale_before,
                        )

                    inner_scaler.update()
                else:
                    # BF16 path: direct step (no scaler needed)
                    inner_optimizer.step()
                    logger.info(
                        "Optimizer step applied (BF16)",
                        grad_norm=float(grad_norm),
                    )

                scheduler.step()
                
                # CRITICAL: Zero gradients AFTER optimizer step (at accumulation boundary)
                inner_optimizer.zero_grad()

                training_time = time.time() - training_start_time
                total_training_time += training_time
                training_start_time = None

                # === Clear memory after optimizer step ===
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Memory cleared after optimizer step")

                # Only compute hash on rank 0 and only occasionally to save memory
                if rank == 0 and step % 100 == 0:  # Only every 100 steps
                    new_model_hash = get_model_hash(model.state_dict())
                    logger.info(f"Updated model", old_model_hash=old_model_hash, new_model_hash=new_model_hash)

            # === Log metric ===
            if (
                is_inner_optimizer_step
                and inner_opt_step % max(round(config.local_par.global_opt_interval * 0.02), 1) == 0
            ):
                logger.info("(2) Logging step", loss_batch=loss_batch, aux_loss_batch=aux_loss_batch)
                metrics = get_status(
                    config=config,
                    model=model,
                    step=step,
                    inner_opt_step=inner_opt_step,
                    training_time=training_time,
                    total_training_time=total_training_time,
                    inner_optimizer=inner_optimizer,
                    loss_batch=loss_batch,
                    aux_loss_batch=aux_loss_batch,
                )
                metric_logger.log(metrics, print_log=False)

            # === local validation and log metric ===
            if False and is_inner_optimizer_step and inner_opt_step % config.log.metric_interval == 0:  # disabled for testing
                logger.info("(3) Local evaluation")

                val_metric = evaluate_model(
                    rank=rank, step=inner_opt_step, model=model, eval_dataloader=train_dataloader, device=device,
                    max_eval_batches=5,  # reduced for faster local eval
                )

                metrics = (
                    get_status(
                        config=config,
                        model=model,
                        step=step,
                        inner_opt_step=inner_opt_step,
                        training_time=training_time,
                        total_training_time=total_training_time,
                        inner_optimizer=inner_optimizer,
                        loss_batch=loss_batch,
                        aux_loss_batch=aux_loss_batch,
                    )
                    | val_metric
                )

                metric_logger.log(metrics)

                logger.info("reached barrier, waiting for partial validation and metric logging to complete")
                # dist.barrier(device_ids=[rank])

            # === save checkpoint ===
            if (
                is_inner_optimizer_step
                and config.ckpt.checkpoint_interval is not None
                and inner_opt_step % config.ckpt.checkpoint_interval == 0
            ):
                logger.info("(4) Saving checkpoint")

                # Only save checkpoint on rank 0 to avoid OOM from multiple processes saving
                if rank == 0:
                    ckpt_path = os.path.join(
                        config.ckpt.checkpoint_path,
                        f"globalver_{current_model_meta.global_ver}_inneropt_{inner_opt_step}",
                    )

                    save_checkpoint(
                        checkpoint_path=ckpt_path,
                        model=model,
                        inner_optimizer=inner_optimizer,
                        scheduler=scheduler,
                        loss=loss_batch.item(),
                        inner_scaler=inner_scaler,
                        data_loader=train_dataloader,
                        save_global_state=True,  # Always True since we're only on rank 0
                        rank=rank,
                        save_model_by_expert_group=True,
                        expert_manager=expert_manager,
                    )

                if config.ckpt.checkpoint_topk is not None:
                    ckpt_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, config.ckpt.checkpoint_topk)
                    if ckpt_deleted:
                        logger.info(f"Deleted old checkpoints: {ckpt_deleted}")

                logger.info("reached barrier, waiting for complete checkpoint saving")
                # dist.barrier(device_ids=[rank])

            # === reload model ===
            # Only check for model reloads at checkpoint intervals, not every step
            # This prevents constantly reloading old checkpoints and wiping out progress between saves
            if (
                is_inner_optimizer_step
                and config.ckpt.checkpoint_interval is not None
                and inner_opt_step % config.ckpt.checkpoint_interval == 0
            ):
                logger.info("(5) Reload Model")

                newest_checkpoint = start_model_from(
                    rank,
                    config,
                    primary_ckpt_path=config.ckpt.validator_checkpoint_path,
                    secondary_ckpt_path=config.ckpt.checkpoint_path,
                )[1]

                if newest_checkpoint > current_model_meta:
                    logger.info(
                        "Should reload model",
                        newest_checkpoint=newest_checkpoint,
                        current_model_meta=current_model_meta,
                    )
                    # dist.barrier(device_ids=[rank])  # make sure everything is saved and everyone is ready to load
                    logger.info("freeing cuda memory")
                    free_cuda_models(models=[model], optimizers=[inner_optimizer], devices=[device])
                    logger.info(
                        "restarting model",
                        current_model_meta=current_model_meta,
                        largest_avail_model=start_model_from(
                            rank,
                            config,
                            primary_ckpt_path=config.ckpt.validator_checkpoint_path,
                            secondary_ckpt_path=config.ckpt.checkpoint_path,
                        )[1],
                    )
                    (
                        model,
                        inner_optimizer,
                        inner_scaler,
                        scheduler,
                        expert_manager,
                        train_dataloader,
                        current_model_version,
                        use_bf16,  # Also update use_bf16 flag on reload
                    ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta)
                    # Update current_model_meta after reload
                    current_model_meta = current_model_version
                    # Update use_fp16_amp based on new use_bf16 flag
                    use_fp16_amp = precision == "fp16-mixed" and torch.cuda.is_available() and not use_bf16
                else:
                    logger.info(
                        "No need to reload model",
                        newest_checkpoint=newest_checkpoint,
                        current_model_meta=current_model_meta,
                    )

            # === Clean up ===
            if is_inner_optimizer_step:
                logger.info("(6) Clean up")
                loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
                aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Clean up completed")

    except Exception:
        logger.error("Quit training", exc_info=True)
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metric_logger.close()

        if rank == 0:
            torch.save(model.state_dict(), "mycelia_final.pt")


def run_distributed_training() -> None:
    """
    Runs the distributed training process.

    Returns:
        None
    """
    args = parse_args()

    if args.path:
        config = MinerConfig.from_path(args.path)
    else:
        config = MinerConfig()

    mp.spawn(
        init_process,
        args=(config, config.local_par.world_size, train_worker),
        nprocs=config.local_par.world_size,
    )


if __name__ == "__main__":
    run_distributed_training()
