"""
Startup self-tests for decentralized consensus system.

These tests verify that the node's environment supports all critical
consensus features before starting validator/miner operations.
"""

import bittensor

from mycelia.shared.app_logging import structlog
from mycelia.shared.chain import read_commitment, read_commitment_at_block
from mycelia.shared.config import WorkerConfig

logger = structlog.get_logger(__name__)


def test_historical_reads_support(
    subtensor: bittensor.Subtensor,
    netuid: int,
    wallet: bittensor.Wallet,
) -> bool:
    """
    Test if substrate client supports historical reads.

    This is CRITICAL for 2-phase commit verification to work.
    If historical reads aren't supported, fail fast with clear error.

    Args:
        subtensor: Bittensor subtensor instance
        netuid: Network UID
        wallet: Wallet to test with

    Returns:
        True if historical reads are supported, False otherwise

    Raises:
        RuntimeError if historical reads are not supported
    """
    logger.info("🔍 Testing historical reads support...")

    try:
        # Get current block
        current_block = subtensor.block
        current_block_hash = subtensor.get_block_hash(current_block)

        logger.debug(
            "Testing historical read",
            block=current_block,
            block_hash=current_block_hash[:16] + "...",
        )

        # Try to read our own commitment at current block (should work if historical reads supported)
        hotkey = wallet.hotkey.ss58_address
        commit_at_block = read_commitment_at_block(
            subtensor=subtensor,
            hotkey=hotkey,
            netuid=netuid,
            block_hash=current_block_hash,
        )

        # Also read current state (for comparison)
        commit_current = read_commitment(
            subtensor=subtensor,
            hotkey=hotkey,
            netuid=netuid,
        )

        # Check if substrate client supports historical queries
        if not hasattr(subtensor, 'substrate') or subtensor.substrate is None:
            logger.error(
                "❌ CRITICAL: Subtensor has no substrate client. "
                "Historical reads are NOT supported. "
                "2-phase commit verification will NOT work."
            )
            raise RuntimeError(
                "Substrate client not available - cannot perform historical reads. "
                "Either upgrade to a substrate-enabled bittensor client or switch to single-phase commits."
            )

        # If commit_at_block is None but commit_current exists, historical reads failed
        if commit_current is not None and commit_at_block is None:
            logger.error(
                "❌ CRITICAL: Historical read returned None but current read succeeded. "
                "Substrate client does not support historical queries. "
                "2-phase commit verification will NOT work.",
                current_commit=commit_current is not None,
                historical_commit=commit_at_block is not None,
            )
            raise RuntimeError(
                "Historical reads not supported by substrate client. "
                "2-phase commit verification requires historical state queries. "
                "Either upgrade substrate client or switch to single-phase signed commits."
            )

        # Both None is OK (no commitment yet)
        # Both exist is OK (historical reads work)
        logger.info(
            "✅ Historical reads are supported",
            has_current_commit=commit_current is not None,
            has_historical_commit=commit_at_block is not None,
        )
        return True

    except RuntimeError:
        # Re-raise RuntimeError (our explicit failures)
        raise

    except Exception as e:
        logger.error(
            "❌ Historical reads self-test failed with exception",
            error=str(e),
            exc_info=True,
        )
        raise RuntimeError(
            f"Historical reads self-test failed: {e}. "
            "Cannot proceed without historical read support."
        )


def test_config_consistency(config: WorkerConfig) -> bool:
    """
    Test that config contains all consensus-critical parameters.

    All nodes MUST have identical values for these parameters, or consensus will fork.

    Args:
        config: Worker configuration

    Returns:
        True if config is valid

    Raises:
        ValueError if config is missing critical parameters
    """
    logger.info("🔍 Testing config consistency...")

    # Check cycle_length
    if not hasattr(config.cycle, 'cycle_length') or config.cycle.cycle_length <= 0:
        raise ValueError("Config missing or invalid cycle_length - consensus-critical parameter")

    # Check phase periods
    required_periods = [
        'distribute_period',
        'train_period',
        'commit_period',
        'submission_period',
        'validate_period',
        'merge_period',
    ]

    for period_name in required_periods:
        if not hasattr(config.cycle, period_name):
            raise ValueError(f"Config missing {period_name} - consensus-critical parameter")

        period_value = getattr(config.cycle, period_name)
        if not isinstance(period_value, int) or period_value < 0:
            raise ValueError(f"Config {period_name} must be non-negative integer, got {period_value}")

    # Verify phase periods sum to cycle length
    from mycelia.shared.chain import get_phase_periods_from_config

    phase_periods = get_phase_periods_from_config(config)
    total_period = sum(phase_periods)

    if total_period != config.cycle.cycle_length:
        logger.error(
            "❌ CRITICAL: Phase periods do not sum to cycle_length",
            cycle_length=config.cycle.cycle_length,
            phase_periods=phase_periods,
            total_period=total_period,
        )
        raise ValueError(
            f"Phase periods sum to {total_period} but cycle_length is {config.cycle.cycle_length}. "
            "These must match for consensus to work."
        )

    logger.warning(
        "⚠️  CONSENSUS-CRITICAL CONFIG - ALL NODES MUST HAVE IDENTICAL VALUES",
        cycle_length=config.cycle.cycle_length,
        phase_periods=phase_periods,
    )
    logger.info("✅ Config consistency check passed")
    return True


def run_all_startup_tests(
    config: WorkerConfig,
    subtensor: bittensor.Subtensor,
    wallet: bittensor.Wallet,
) -> bool:
    """
    Run all startup self-tests.

    Args:
        config: Worker configuration
        subtensor: Bittensor subtensor instance
        wallet: Wallet for testing

    Returns:
        True if all tests pass

    Raises:
        RuntimeError or ValueError if any critical test fails
    """
    logger.info("🚀 Running startup self-tests...")

    # Test 1: Config consistency
    test_config_consistency(config)

    # Test 2: Historical reads support
    test_historical_reads_support(
        subtensor=subtensor,
        netuid=config.chain.netuid,
        wallet=wallet,
    )

    logger.info("✅ All startup self-tests passed")
    return True
