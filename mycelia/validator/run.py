import asyncio
import copy
import gc
import os
import secrets
from typing import Any

import bittensor
import torch
import torch.nn as nn
from hivemind.averaging import DecentralizedAverager
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from mycelia.miner.train_helper import get_status
from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import (
    ValidatorChainCommit,
    commit_status,
    get_consensus_miner_list,
    get_deterministic_seed,
    setup_chain_worker,
    submit_weights_to_chain,
    verify_validator_consensus_on_model,
)
from mycelia.shared.checkpoint import (
    ModelMeta,
    compile_full_state_dict_from_path,
    delete_old_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from mycelia.shared.config import ValidatorConfig, parse_args
from mycelia.shared.cycle import gather_validation_job, get_combined_validator_seed, wait_till
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.evaluate import evaluate_model
from mycelia.shared.expert_manager import (
    ExpertManager,
    get_weight_sum,
    populate_global_grads_from_local,
)
from mycelia.shared.helper import get_model_hash, get_nested_attr
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.model import load_model
from mycelia.shared.modeling.mycelia import get_base_tokenizer
from mycelia.sn_owner.cycle import PhaseNames
from mycelia.validator.aggregator import MinerScoreAggregator
from mycelia.validator.evaluator import (
    MinerEvalJob,
    load_model_from_path,
    run_evaluation,
)
from mycelia.validator.inter_validator_connection import (
    build_averagers_from_buff,
    build_grad_buff_from_model,
    connect_with_peers,
    pack_grads,
    unpack_to_grads,
)

configure_logging()
logger = structlog.get_logger(__name__)


def cleanup(global_model, base_model) -> None:
    """
    Cleans up the distributed training environment.

    Returns:
        None
    """
    torch.cuda.synchronize()
    global_model.to("cpu")
    base_model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


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
    torch.nn.Module,  # global_model
    torch.optim.Optimizer,  # outer_optimizer
    torch.amp.GradScaler,  # outer_scaler
    int,  # start_step
    "ExpertManager",  # em
    StatefulDataLoader,
]:
    """
    Build model(s), experts layout, optimizers, scheduler, scaler, and optionally resume from a checkpoint.
    """
    # === checkpoint info ===
    resume = False
    latest_checkpoint_path = None

    # === model & Experts manager ===
    logger.info(f"setup training - load model and expert manager")
    expert_manager = ExpertManager(config)
    base_model, model_meta = load_model(rank, config, expert_manager, subtensor, wallet, current_model_meta)
    base_model = base_model.cpu()
    global_model = copy.deepcopy(base_model)

    # === optimizers ===
    logger.info(f"setup training - load optimizer")
    outer_optimizer = torch.optim.SGD(
        global_model.named_parameters(),
        lr=config.opt.outer_lr,
        momentum=config.opt.outer_momentum,
        nesterov=True,
    )

    # === scaler ===
    logger.info(f"setup training - load scaler")
    outer_scaler = torch.amp.GradScaler(
        "cuda", enabled=(get_nested_attr(config, "model.precision", "") == "fp16-mixed")
    )

    # === dataloader ===
    logger.info(f"setup training - load dataloader")
    train_dataloader = get_dataloader(config, rank=rank, world_size=config.task.data.world_size, tokenizer=tokenizer)

    # === load checkpoint (if any) ===
    logger.info(
        f"setup training - load past checkpoint"
    )  # outer_optimizer is static, so dont really need to load checkpoint
    if get_nested_attr(config, "resume_from_ckpt", False) and resume and latest_checkpoint_path:
        _ = load_checkpoint(
            config=config,
            checkpoint_path=latest_checkpoint_path,
            outer_optimizer=outer_optimizer,
            outer_scaler=outer_scaler,
            rank=rank,
            device=device,
            data_loader=train_dataloader,
        )

    logger.info(f"setup_training - completed successfully!")
    return (
        base_model,
        global_model,
        outer_optimizer,
        outer_scaler,
        model_meta.global_ver,
        expert_manager,
        train_dataloader,
    )


async def aggregate_miner_gradient_change(
    base_model: nn.Module,
    global_model: nn.Module,
    device: torch.device,
    rank: int,
    outer_optimizer: torch.optim.Optimizer,
    miner_jobs: list[MinerEvalJob],
    score_aggregator: MinerScoreAggregator,
):
    global_model.to(device)
    
    # Load all miners with positive scores and store their scores
    miner_models_with_scores: dict[str, tuple[nn.Module, float]] = {}
    total_score = 0.0
    
    for miner_job in miner_jobs:
        # Get miner's average score
        score = score_aggregator.avg(uid=miner_job.uid)
        
        # Only include miners with positive scores
        if score > 0:
            miner_model = await asyncio.to_thread(
                load_model_from_path, miner_job.model_path, base_model, device
            )
            miner_models_with_scores[miner_job.uid] = (miner_model, score)
            total_score += score
    
    if not miner_models_with_scores:
        logger.warning("No miners with positive scores to aggregate")
        return
    
    logger.info(
        f"Aggregating {len(miner_models_with_scores)} miners with total score {total_score:.4f}"
    )
    
    # Weight each miner's gradients proportionally to their score
    for uid, (miner_model, score) in miner_models_with_scores.items():
        normalized_weight = score / total_score
        logger.debug(f"Miner {uid[:16]}... weight: {normalized_weight:.4f}")
        populate_global_grads_from_local(global_model, miner_model, weight=normalized_weight)


def sync_grad_across_validators(
    group_averagers: dict[str | int, DecentralizedAverager], group_grad_buff_meta: dict[str | int, Any]
):
    for group_id, avg in group_averagers.items():
        if avg.total_size <= 0:
            logger.info("skip averager", group_id=group_id, mode=avg.mode, total_size=avg.total_size)
            continue

        logger.info("begin sync grad across validator", group=group_id, mode=avg.mode)
        pack_grads(group_grad_buff_meta[group_id])
        info = avg.step(allow_retries=False)
        unpack_to_grads(group_grad_buff_meta[group_id])

        logger.info(
            "sync grad across validator", group=group_id, mode=avg.mode, successes=("averaged" if info else "no group")
        )

    return


def run_global_optimization(
    model: nn.Module,
    global_model: nn.Module,
    device: torch.device,
    rank: int,
    outer_optimizer: torch.optim.Optimizer,
    miner_jobs: list[MinerEvalJob],
    score_aggregator: MinerScoreAggregator,
):
    # --- sync + outer step ---
    # keep global model on device for syncing/stepping, then move back to CPU
    global_model.to(device)

    old_shared_name, old_shared_sum = get_weight_sum(model, shared=True)
    old_expert_name, old_expert_sum = get_weight_sum(model, shared=False)

    logger.info("start syncing shared weights")

    outer_optimizer.step()
    outer_optimizer.zero_grad()

    # copy updated global weights back into the worker model safely
    with torch.no_grad():
        model.load_state_dict(global_model.state_dict(), strict=True)

    new_shared_name, new_shared_sum = get_weight_sum(model, shared=True)
    new_expert_name, new_expert_sum = get_weight_sum(model, shared=False)

    logger.info(
        "outer optimizer step (shared)",
        param_name=old_shared_name,
        old_sum=round(float(old_shared_sum), 6),
        new_sum=round(float(new_shared_sum), 6),
    )
    logger.info(
        "outer optimizer step (expert)",
        param_name=old_expert_name,
        old_sum=round(float(old_expert_sum), 6),
        new_sum=round(float(new_expert_sum), 6),
    )


def run(rank: int, world_size: int, config: ValidatorConfig) -> None:
    """
    The worker function for training in a distributed setting.

    Args:
        rank (int): The rank of the process.
        world_size (int): The total number of processes.
        config (Config): The configuration object for the training.

    Returns:
        None
    """
    if rank == 0:
        config.write()

    # === create checkpoint directory ===
    os.makedirs(config.ckpt.base_checkpoint_path, exist_ok=True)
    os.makedirs(config.ckpt.checkpoint_path, exist_ok=True)
    os.makedirs(config.log.base_metric_path, exist_ok=True)
    os.makedirs(config.ckpt.miner_submission_path, exist_ok=True)

    # === set up chain worker ===
    wallet, subtensor = setup_chain_worker(config)

    # === set logging ===
    metric_logger = MetricLogger(config, rank)

    # === mis ===
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    tokenizer = get_base_tokenizer(config)

    eval_dataloader = get_dataloader(
        config,
        rank=0,
        world_size=10,
        tokenizer=tokenizer,
    )

    # === set up training ===
    (
        base_model,
        global_model,
        outer_optimizer,
        outer_scaler,
        start_step,
        expert_manager,
        train_dataloader,
    ) = setup_training(config, rank, device, tokenizer, subtensor, wallet, current_model_meta=None)

    global_opt_step = start_step

    # === set up score aggregator ===
    score_aggregator = MinerScoreAggregator()

    # === set up averager ===
    group_grad_buff_meta = build_grad_buff_from_model(
        model=base_model, expert_group_assignment=expert_manager.expert_group_assignment
    )

    dht = connect_with_peers()

    group_averagers = build_averagers_from_buff(group_buff_metas=group_grad_buff_meta, dht=dht)

    commit_status(
        config,
        wallet,
        subtensor,
        ValidatorChainCommit(
            model_hash=None,
            global_ver=global_opt_step,
            expert_group=config.task.expert_group_id,
            miner_seed=0,  # this should reveal later
            block=subtensor.block,
        ),
    )

    # === training ===
    loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    aux_loss_batch = torch.tensor(0, dtype=torch.float32, device=device)
    training_time = 0
    total_training_time = 0

    outer_optimizer.zero_grad()

    current_model_hash = None

    try:
        while True:
            # for each step, we run 1 backward
            # for each inner_opt_step, we run local optimization; gradient_accumulation_steps = 1 real step
            # for each global_opt_interval number of inner_opt_step, we synchronise weight from different ddp worker, and then run global optimization

            # === Wait till commit phase to submit deterministic seed ===
            wait_till(config, PhaseNames.commit, subtensor)  # Pass subtensor for decentralized
            logger.info("(0) Commit new seed for next validation")

            # Use deterministic seed from chain for consensus
            # CRITICAL: Use cycle anchor block so ALL validators get SAME seed
            current_block = subtensor.block
            deterministic_seed = get_deterministic_seed(
                subtensor,
                current_block,
                config.chain.netuid,
                cycle_length=config.cycle.cycle_length,  # Use config, not hardcoded 600
            )

            # CRITICAL: Include commit_phase=1 so get_active_validators() recognizes this
            commit_status(
                config,
                wallet,
                subtensor,
                ValidatorChainCommit(
                    model_hash=None,  # No model hash in seed commit (phase 1)
                    global_ver=global_opt_step,
                    expert_group=config.task.expert_group_id,
                    miner_seed=deterministic_seed,
                    commit_phase=1,  # CRITICAL: Phase 1 = seed commit
                    block=current_block,
                ),
            )

            # === Wait till validation phase to start the validation procedure ===
            wait_till(config, PhaseNames.validate, subtensor)  # Pass subtensor for decentralized
            logger.info("(1) Start validation pahse")

            # === Get miner ===
            logger.info("(2) Gathering miner job")
            miner_jobs = gather_validation_job(config, subtensor, step=global_opt_step)
            logger.info("miner job(s)", global_opt_step=global_opt_step, miner_jobs=miner_jobs)

            # === Get miner model and evaluate the miners ===
            logger.info("(3) Evaluating miners with unpredictable validation")
            asyncio.run(
                run_evaluation(
                    config=config,
                    step=global_opt_step,
                    device=device,  # operate at cuda
                    miners=miner_jobs,
                    score_aggregator=score_aggregator,
                    base_model=base_model.to("cpu"),
                    tokenizer=tokenizer,
                    combinded_seed=get_combined_validator_seed(config, subtensor),
                    subtensor=subtensor,  # Pass subtensor for unpredictable validation
                    use_unpredictable=True,  # Enable unpredictable validation
                )
            )

            cleanup(global_model, base_model)
            logger.info("eval result", scores=score_aggregator.uid_score_pairs())

            # === Submit weights to chain ===
            logger.info("(3a) Submitting weights to chain")
            weight_success = submit_weights_to_chain(
                subtensor=subtensor,
                wallet=wallet,
                netuid=config.chain.netuid,
                score_aggregator=score_aggregator,
                min_score_threshold=0.0,  # Include all miners with score > 0
            )
            if not weight_success:
                logger.warning("Failed to submit weights to chain, continuing anyway")

            # === aggragate miner gradient change locally ===
            logger.info("(4) Aggregating miner gradient change")
            asyncio.run(
                aggregate_miner_gradient_change(
                    base_model=base_model.to("cpu"),
                    global_model=global_model.to("cpu"),
                    device=torch.device("cpu"),  # all gradient aggregation done on cpu
                    rank=rank,
                    outer_optimizer=outer_optimizer,
                    miner_jobs=miner_jobs,
                    score_aggregator=score_aggregator,
                )
            )

            # === wait till merging phase and aggragate miner gradient change ===
            wait_till(config, PhaseNames.merge, subtensor)  # Pass subtensor for decentralized
            logger.info("(5) Syncing gradient across validators")
            sync_grad_across_validators(group_averagers=group_averagers, group_grad_buff_meta=group_grad_buff_meta)

            # === global optimizer ===
            logger.info("(6) Running global model optimization step")
            run_global_optimization(
                model=base_model,
                global_model=global_model.to("cpu"),
                device=device,
                rank=rank,
                outer_optimizer=outer_optimizer,
                miner_jobs=miner_jobs,
                score_aggregator=score_aggregator,
            )

            cleanup(global_model, base_model)

            # === save checkpoint ===
            logger.info("(7) Saving checkpoint")
            ckpt_path = config.ckpt.checkpoint_path / f"globalver_{int(global_opt_step)}"

            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=base_model,
                outer_optimizer=outer_optimizer,
                loss=loss_batch.item(),
                outer_scaler=outer_scaler,
                data_loader=train_dataloader,
                save_global_state=rank == 0,
                rank=rank,
                expert_manager=expert_manager,
                save_model_by_expert_group=True,
            )

            # === Commit model to chain with signature ===
            current_model_hash = get_model_hash(compile_full_state_dict_from_path(ckpt_path)).hex()
            current_block = subtensor.block

            # Get anchor block for domain-separated signature
            from mycelia.shared.chain import get_cycle_anchor_block
            anchor_block = get_cycle_anchor_block(current_block, config.cycle.cycle_length)
            anchor_block_hash = subtensor.get_block_hash(anchor_block)

            # Domain-separated message: mycelia:v1|netuid|expert_group|anchor_hash|phase|model_hash
            # Use phase=2 for model distribution phase (phase 1 was seed commit)
            domain_message = f"mycelia:v1|{config.chain.netuid}|{config.task.expert_group_id}|{anchor_block_hash}|2|{current_model_hash}"
            signed_hash = wallet.hotkey.sign(domain_message.encode()).hex()

            logger.info("(7a) Committing model with signature to chain")
            commit_status(
                config,
                wallet,
                subtensor,
                ValidatorChainCommit(
                    model_hash=current_model_hash,
                    signed_model_hash=signed_hash,  # Signature over domain-separated message
                    global_ver=global_opt_step,
                    expert_group=config.task.expert_group_id,
                    miner_seed=0,  # No seed in model commit
                    commit_phase=2,  # Phase 2 = model distribution
                    block=current_block,
                ),
            )

            if config.ckpt.checkpoint_topk is not None:
                ckpt_deleted = delete_old_checkpoints(config.ckpt.checkpoint_path, config.ckpt.checkpoint_topk)
                if ckpt_deleted:
                    logger.info(f"Deleted old checkpoints: {ckpt_deleted}")

            # === validation and log metric ===
            logger.info("(8) Start local evaluation")
            val_metric = evaluate_model(
                rank=rank,
                step=global_opt_step,
                model=global_model.to("cpu"),
                eval_dataloader=eval_dataloader,
                device=device,
            )

            metrics = (
                get_status(
                    config=config,
                    model=base_model,
                    step=global_opt_step,
                    training_time=training_time,
                    total_training_time=total_training_time,
                    inner_opt_step=None,
                    global_opt_step=global_opt_step,
                    loss_batch=loss_batch,
                    aux_loss_batch=aux_loss_batch,
                )
                | val_metric
            )

            metric_logger.log(metrics)
            cleanup(global_model, base_model)

            # === Clean up ===
            global_opt_step += 1

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received, shutting down validator loop")
        cleanup()
        metric_logger.close()
        for _, a in group_averagers.items():
            a.shutdown()
        raise
    except Exception:
        logger.error("Quit training", exc_info=True)
        cleanup()
        metric_logger.close()
        for _, a in group_averagers.items():
            a.shutdown()

        if rank == 0:
            torch.save(global_model.state_dict(), "mycelia_final.pt")


if __name__ == "__main__":
    args = parse_args()

    if args.path:
        config = ValidatorConfig.from_path(args.path)
    else:
        config = ValidatorConfig()

    run(0, 1, config)
