import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue
from threading import Lock, Thread

import bittensor

from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import MinerChainCommit, _subtensor_lock, commit_status
from mycelia.shared.checkpoint import (
    ModelMeta,
    compile_full_state_dict_from_path,
    delete_old_checkpoints,
    get_resume_info,
    start_model_from,
)
from mycelia.shared.circuit_breaker import RetryWithBackoff
from mycelia.shared.client import submit_model
from mycelia.shared.config import MinerConfig, parse_args
from mycelia.shared.cycle import search_model_submission_destination, setup_chain_worker, wait_till
from mycelia.shared.fork_safe_download import fetch_model_from_chain_safe
from mycelia.shared.helper import get_model_hash
from mycelia.shared.schema import sign_message
from mycelia.sn_owner.cycle import PhaseNames

configure_logging()
logger = structlog.get_logger(__name__)


# --- Job definitions ---


class JobType(Enum):
    DOWNLOAD = auto()
    SUBMIT = auto()
    COMMIT = auto()


@dataclass
class Job:
    job_type: JobType
    payload: dict | None = None
    phase_end_block: int | None = None


@dataclass
class SharedState:
    current_model_version: int | None = None
    current_model_hash: str | None = None
    latest_checkpoint_path: str | None = None
    lock: Lock = field(default_factory=Lock, repr=False)


class FileNotReadyError(RuntimeError):
    pass


# --- Scheduler service ---
def scheduler_service(
    config,
    download_queue: Queue,
    commit_queue: Queue,
    submit_queue: Queue,
    poll_fallback_block: int = 3,
):
    """
    Periodically checks whether to start download/submit phases and enqueues jobs.
    """
    while True:
        # --------- DOWNLOAD SCHEDULING ---------
        wait_till(config, phase_name=PhaseNames.distribute, poll_fallback_block=poll_fallback_block)
        download_queue.put(Job(job_type=JobType.DOWNLOAD))

        # --------- COMISSION SCHEDULING ---------
        _, phase_end_block = wait_till(config, phase_name=PhaseNames.commit, poll_fallback_block=poll_fallback_block)
        commit_queue.put(Job(job_type=JobType.COMMIT, phase_end_block=phase_end_block))

        # --------- SUBMISSION SCHEDULING ---------
        wait_till(config, phase_name=PhaseNames.submission, poll_fallback_block=poll_fallback_block)
        submit_queue.put(Job(job_type=JobType.SUBMIT))


# --- Workers ---
def download_worker(
    config,
    download_queue: Queue,
    wallet,
    current_model_meta,
    current_model_hash,
    shared_state: SharedState,
):
    """
    Consumes DOWNLOAD jobs and runs the download phase logic.

    FORK-SAFE: Only downloads if validators reach consensus on model hash.
    """
    subtensor = bittensor.Subtensor(config.chain.network)
    while True:
        job = download_queue.get()
        try:
            # Read current version/hash snapshot
            _, current_model_meta, _ = start_model_from(
                0,
                config,
                primary_ckpt_path=config.ckpt.validator_checkpoint_path,
                secondary_ckpt_path=config.ckpt.checkpoint_path,
            )

            current_model_meta.model_hash = current_model_hash

            # FORK-SAFE DOWNLOAD: Check validator consensus before downloading
            download_meta = fetch_model_from_chain_safe(
                current_model_meta=current_model_meta,
                config=config,
                subtensor=subtensor,
                wallet=wallet,
            )

            # If no consensus or no new model, skip download
            if download_meta is None:
                logger.info(
                    f"<{PhaseNames.distribute}> No consensus or no new model - "
                    "staying on current model (fork-safe behavior)"
                )
                continue

            if (
                not isinstance(download_meta, dict)
                or "model_hash" not in download_meta
            ):
                raise FileNotReadyError(f"Invalid download metadata: {download_meta}")

            logger.info(
                f"<{PhaseNames.distribute}> ✅ Downloaded model with consensus - "
                f"hash: {download_meta['model_hash'][:16]}..., "
                f"validators agreeing: {download_meta.get('agreeing_validators', '?')}"
            )

            # Update shared state with new version/hash
            _, current_model_meta, _ = start_model_from(
                0,
                config,
                primary_ckpt_path=config.ckpt.validator_checkpoint_path,
                secondary_ckpt_path=config.ckpt.checkpoint_path,
            )

            with shared_state.lock:
                shared_state.current_model_version = current_model_meta.global_ver
                shared_state.current_model_hash = download_meta["model_hash"]

        except FileNotReadyError as e:
            logger.warning(f"<{PhaseNames.distribute}> File not ready error: {e}")

        except Exception as e:
            logger.error(f"<{PhaseNames.distribute}> Error while handling job: {e}")
            traceback.print_exc()

        finally:
            download_queue.task_done()
            logger.info(f"<{PhaseNames.distribute}> task completed.")


def commit_worker(
    config,
    commit_queue: Queue,
    wallet,
    shared_state: SharedState,
):
    """
    Consumes COMMIT model and runs the submission phase logic.
    """
    subtensor = bittensor.Subtensor(config.chain.network)
    while True:
        job = commit_queue.get()
        try:
            _, model_meta, latest_checkpoint_path = get_resume_info(rank=0, config=config)

            with shared_state.lock:
                shared_state.latest_checkpoint_path = latest_checkpoint_path

            if latest_checkpoint_path is None:
                raise FileNotReadyError("Not checkpoint found, skip commit.")

            model_path = f"{latest_checkpoint_path}/model.pt"
            model_hash = get_model_hash(compile_full_state_dict_from_path(model_path)).hex()

            logger.info(
                f"<{PhaseNames.commit}> committing model version {model_meta.global_ver} with hash {model_hash}."
            )
            commit_status(
                config,
                wallet,
                subtensor,
                MinerChainCommit(
                    expert_group=config.task.expert_group_id,
                    model_hash=model_hash,
                    block=subtensor.block,
                    global_ver=model_meta.global_ver,
                    inner_opt=model_meta.inner_opt,
                ),
            )

        except FileNotReadyError as e:
            logger.warning(f"<{PhaseNames.commit}> File not ready error: {e}")

        except Exception as e:
            logger.error(f"<{PhaseNames.commit}> Error while handling job: {e}")
            traceback.print_exc()

        finally:
            commit_queue.task_done()
            logger.info(f"<{PhaseNames.commit}> task completed.")


def submit_worker(
    config,
    submit_queue: Queue,
    wallet,
    shared_state: SharedState,
):
    """
    Consumes SUBMIT jobs and runs the submission phase logic.
    """
    subtensor = bittensor.Subtensor(config.chain.network)
    max_retries = 3
    retry_delay = 30  # seconds
    
    while True:
        job = submit_queue.get()

        try:
            with shared_state.lock:
                latest_checkpoint_path = shared_state.latest_checkpoint_path

            if latest_checkpoint_path is None:
                raise FileNotReadyError("Not checkpoint found, skip submission.")

            # Retry logic for finding validator
            destination_axon = None
            for attempt in range(max_retries):
                destination_axon = search_model_submission_destination(
                    wallet=wallet,
                    config=config,
                    subtensor=subtensor,
                )
                
                if destination_axon is not None:
                    break
                
                if attempt < max_retries - 1:
                    logger.warning(
                        f"<{PhaseNames.submission}> No validator destination found (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"<{PhaseNames.submission}> Failed to find validator destination after {max_retries} attempts. "
                        "This miner may not be assigned to any validator."
                    )
            
            if destination_axon is None:
                logger.error(f"<{PhaseNames.submission}> Skipping submission - no validator available")
                continue
                
            block = subtensor.block

            # Retry logic for submission
            submission_success = False
            for attempt in range(max_retries):
                try:
                    submit_model(
                        url=f"http://{destination_axon.ip}:{destination_axon.port}/submit-checkpoint",
                        token="",
                        my_hotkey=wallet.hotkey,  # type: ignore
                        target_hotkey_ss58=destination_axon.hotkey,
                        block=block,
                        model_path=f"{latest_checkpoint_path}/model_expgroup_{config.task.expert_group_id}.pt",
                    )
                    submission_success = True
                    logger.info(
                        f"<{PhaseNames.submission}> submitted model to {destination_axon.hotkey} "
                        f"at block {block}."
                    )
                    break
                except Exception as submit_err:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"<{PhaseNames.submission}> Submission failed (attempt {attempt + 1}/{max_retries}): {submit_err}, "
                            f"retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"<{PhaseNames.submission}> Submission failed after {max_retries} attempts: {submit_err}"
                        )
                        traceback.print_exc()

        except FileNotReadyError as e:
            logger.warning(f"<{PhaseNames.submission}> File not ready error: {e}")

        except Exception as e:
            logger.error(f"<{PhaseNames.submission}> Error while handling job: {e}")
            traceback.print_exc()

        finally:
            submit_queue.task_done()
            logger.info(f"<{PhaseNames.submission}> task completed.")


# --- Wiring it all together ---
def run_system(config, wallet, current_model_version: int = 0, current_model_hash: str = "xxx"):
    download_queue = Queue()
    commit_queue = Queue()
    submit_queue = Queue()
    shared_state = SharedState(current_model_version, current_model_hash)

    # Start workers
    Thread(
        target=download_worker,
        args=(config, download_queue, wallet, current_model_version, current_model_hash, shared_state),
        daemon=True,
    ).start()

    Thread(
        target=commit_worker,
        args=(config, commit_queue, wallet, shared_state),
        daemon=True,
    ).start()

    Thread(
        target=submit_worker,
        args=(config, submit_queue, wallet, shared_state),
        daemon=True,
    ).start()

    # Start scheduler (runs in foreground)
    scheduler_service(
        config=config,
        download_queue=download_queue,
        commit_queue=commit_queue,
        submit_queue=submit_queue,
    )


if __name__ == "__main__":
    args = parse_args()

    if args.path:
        config = MinerConfig.from_path(args.path)
    else:
        config = MinerConfig()

    config.write()

    wallet, _ = setup_chain_worker(config)

    run_system(config, wallet)
