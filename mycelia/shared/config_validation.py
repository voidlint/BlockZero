"""
Configuration validation for consensus-critical parameters.

All nodes in the subnet MUST have identical config values for these parameters,
or consensus will fork.
"""

import hashlib
import json

from mycelia.shared.app_logging import structlog
from mycelia.shared.chain import get_phase_periods_from_config
from mycelia.shared.config import WorkerConfig

logger = structlog.get_logger(__name__)


def compute_config_hash(config: WorkerConfig) -> str:
    """
    Compute hash of consensus-critical config parameters.

    This hash should be identical across all nodes. If any node has a different
    hash, consensus will fork.

    Args:
        config: Worker configuration

    Returns:
        SHA256 hash of consensus-critical parameters
    """
    # Extract consensus-critical parameters
    consensus_params = {
        "cycle_length": config.cycle.cycle_length,
        "phase_periods": get_phase_periods_from_config(config),
        "netuid": config.chain.netuid,
    }

    # Deterministic JSON encoding
    params_json = json.dumps(consensus_params, sort_keys=True)
    config_hash = hashlib.sha256(params_json.encode()).hexdigest()

    return config_hash


def validate_config_consensus_safety(config: WorkerConfig) -> bool:
    """
    Validate that config has all consensus-critical parameters and they're valid.

    Args:
        config: Worker configuration

    Returns:
        True if config is valid

    Raises:
        ValueError if config is invalid or missing critical parameters
    """
    # Check cycle_length
    if not hasattr(config.cycle, 'cycle_length'):
        raise ValueError("Config missing cycle_length - consensus-critical parameter")

    if config.cycle.cycle_length <= 0:
        raise ValueError(f"cycle_length must be positive, got {config.cycle.cycle_length}")

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
            raise ValueError(f"{period_name} must be non-negative integer, got {period_value}")

    # Verify phase periods sum to cycle length
    phase_periods = get_phase_periods_from_config(config)
    total_period = sum(phase_periods)

    if total_period != config.cycle.cycle_length:
        raise ValueError(
            f"Phase periods sum to {total_period} but cycle_length is {config.cycle.cycle_length}. "
            "These must match for consensus to work."
        )

    # Check netuid
    if not hasattr(config.chain, 'netuid'):
        raise ValueError("Config missing netuid - consensus-critical parameter")

    if config.chain.netuid < 0:
        raise ValueError(f"netuid must be non-negative, got {config.chain.netuid}")

    logger.info("✅ Config consensus-safety validation passed")
    return True


def log_consensus_critical_config(config: WorkerConfig):
    """
    Log consensus-critical config parameters prominently.

    This warning helps operators verify that all nodes have identical config.

    Args:
        config: Worker configuration
    """
    phase_periods = get_phase_periods_from_config(config)
    config_hash = compute_config_hash(config)

    logger.warning(
        "\n" + "=" * 80 + "\n"
        "⚠️  CONSENSUS-CRITICAL CONFIGURATION\n"
        "=" * 80 + "\n"
        "ALL NODES MUST HAVE IDENTICAL VALUES FOR THESE PARAMETERS\n"
        "IF ANY NODE HAS DIFFERENT VALUES → CONSENSUS FORK\n"
        "\n"
        f"  cycle_length:       {config.cycle.cycle_length}\n"
        f"  phase_periods:      {phase_periods}\n"
        f"    - distribute:     {phase_periods[0]}\n"
        f"    - train:          {phase_periods[1]}\n"
        f"    - commit:         {phase_periods[2]}\n"
        f"    - submission:     {phase_periods[3]}\n"
        f"    - validate:       {phase_periods[4]}\n"
        f"    - merge:          {phase_periods[5]}\n"
        f"  netuid:             {config.chain.netuid}\n"
        "\n"
        f"  CONFIG HASH:        {config_hash}\n"
        "=" * 80 + "\n"
        "Verify this hash matches on ALL validators and miners.\n"
        "=" * 80
    )


def compare_config_with_expected(config: WorkerConfig, expected_hash: str) -> bool:
    """
    Compare config hash with expected hash.

    Args:
        config: Worker configuration
        expected_hash: Expected config hash

    Returns:
        True if config matches expected hash

    Raises:
        ValueError if config doesn't match expected hash
    """
    actual_hash = compute_config_hash(config)

    if actual_hash != expected_hash:
        phase_periods = get_phase_periods_from_config(config)
        raise ValueError(
            f"Config hash mismatch!\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}\n"
            f"\n"
            f"Your config:\n"
            f"  cycle_length:  {config.cycle.cycle_length}\n"
            f"  phase_periods: {phase_periods}\n"
            f"  netuid:        {config.chain.netuid}\n"
            f"\n"
            f"This node will NOT achieve consensus with the rest of the subnet."
        )

    logger.info(
        "✅ Config hash matches expected value",
        config_hash=actual_hash,
    )
    return True


# ================================================================================
# Known good config hashes (for specific subnet deployments)
# ================================================================================

# Example: If you're deploying to a specific subnet, add the expected config hash here
# SUBNET_1_EXPECTED_HASH = "abc123..."

# def validate_subnet_1_config(config: WorkerConfig) -> bool:
#     """Validate config for subnet 1"""
#     return compare_config_with_expected(config, SUBNET_1_EXPECTED_HASH)
