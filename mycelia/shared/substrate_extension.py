"""
Substrate client extension for historical state queries.

This module extends the bittensor substrate client to support historical reads
at specific block hashes, which is critical for 2-phase commit verification.
"""

import bittensor
from substrateinterface import SubstrateInterface

from mycelia.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


def ensure_substrate_client(subtensor: bittensor.Subtensor) -> SubstrateInterface:
    """
    Ensure subtensor has a substrate client for historical queries.

    Args:
        subtensor: Bittensor subtensor instance

    Returns:
        SubstrateInterface client

    Raises:
        RuntimeError if substrate client is not available
    """
    if not hasattr(subtensor, 'substrate') or subtensor.substrate is None:
        raise RuntimeError(
            "Subtensor does not have substrate client. "
            "Historical reads require substrate client support."
        )

    return subtensor.substrate


def query_historical_commitment(
    subtensor: bittensor.Subtensor,
    hotkey: str,
    netuid: int,
    block_hash: str,
) -> dict | None:
    """
    Query commitment from chain at a specific block hash.

    This is the low-level implementation of historical state reads.

    Args:
        subtensor: Bittensor subtensor instance
        hotkey: Hotkey to query commitment for
        netuid: Network UID
        block_hash: Block hash to query at

    Returns:
        Commitment dict or None if not found

    Raises:
        RuntimeError if substrate client doesn't support historical queries
    """
    try:
        substrate = ensure_substrate_client(subtensor)

        # Query the Commitments pallet at specific block
        # Storage key: Commitments.CommitmentOf(netuid, hotkey)
        result = substrate.query(
            module='Commitments',
            storage_function='CommitmentOf',
            params=[netuid, hotkey],
            block_hash=block_hash,
        )

        if result is None or (hasattr(result, 'value') and result.value is None):
            return None

        # Extract value
        if hasattr(result, 'value'):
            commit_data = result.value
        else:
            commit_data = result

        # Parse commitment data
        if isinstance(commit_data, bytes):
            commit_data = commit_data.decode('utf-8')

        import json
        commit_dict = json.loads(commit_data)

        return commit_dict

    except AttributeError as e:
        raise RuntimeError(
            f"Substrate client does not support historical queries: {e}. "
            "Upgrade to a substrate client that supports block_hash parameter in query()."
        )

    except Exception as e:
        logger.error(
            "Error querying historical commitment",
            hotkey=hotkey[:16] + "...",
            block_hash=block_hash[:16] + "...",
            error=str(e),
        )
        raise


def query_multiple_historical_commitments(
    subtensor: bittensor.Subtensor,
    hotkeys: list[str],
    netuid: int,
    block_hash: str,
) -> dict[str, dict]:
    """
    Query multiple commitments at a specific block hash (batch operation).

    Args:
        subtensor: Bittensor subtensor instance
        hotkeys: List of hotkeys to query
        netuid: Network UID
        block_hash: Block hash to query at

    Returns:
        Dict mapping hotkey → commitment_dict (only for hotkeys with commitments)
    """
    results = {}

    for hotkey in hotkeys:
        try:
            commit = query_historical_commitment(
                subtensor=subtensor,
                hotkey=hotkey,
                netuid=netuid,
                block_hash=block_hash,
            )

            if commit is not None:
                results[hotkey] = commit

        except Exception as e:
            logger.warning(
                f"Failed to query commitment for {hotkey[:16]}...: {e}",
                exc_info=False,
            )
            continue

    return results


def verify_historical_read_capability(subtensor: bittensor.Subtensor) -> bool:
    """
    Verify that the substrate client supports historical reads.

    Args:
        subtensor: Bittensor subtensor instance

    Returns:
        True if historical reads are supported, False otherwise
    """
    try:
        substrate = ensure_substrate_client(subtensor)

        # Check if substrate.query supports block_hash parameter
        import inspect
        sig = inspect.signature(substrate.query)

        if 'block_hash' not in sig.parameters:
            logger.error(
                "Substrate client query() method does not accept block_hash parameter. "
                "Historical reads are NOT supported."
            )
            return False

        logger.info("✓ Substrate client supports historical reads (block_hash parameter)")
        return True

    except RuntimeError:
        logger.error("Substrate client not available")
        return False

    except Exception as e:
        logger.error(f"Error checking historical read capability: {e}")
        return False


def get_storage_map_keys_at_block(
    subtensor: bittensor.Subtensor,
    module: str,
    storage_function: str,
    block_hash: str,
) -> list:
    """
    Get all keys from a storage map at a specific block.

    Useful for getting all commitments at a specific block.

    Args:
        subtensor: Bittensor subtensor instance
        module: Pallet module name (e.g., "Commitments")
        storage_function: Storage function name (e.g., "CommitmentOf")
        block_hash: Block hash to query at

    Returns:
        List of keys in the storage map
    """
    try:
        substrate = ensure_substrate_client(subtensor)

        # Query storage map keys at specific block
        keys = substrate.query_map(
            module=module,
            storage_function=storage_function,
            block_hash=block_hash,
        )

        return list(keys)

    except AttributeError as e:
        raise RuntimeError(
            f"Substrate client does not support historical query_map: {e}"
        )

    except Exception as e:
        logger.error(f"Error querying storage map keys: {e}")
        raise


# ================================================================================
# Alternative implementation if substrate client doesn't support historical reads
# ================================================================================

class HistoricalStateCache:
    """
    Cache for historical state if substrate doesn't support native historical queries.

    This is a fallback mechanism that stores commitments at each block locally.
    NOT RECOMMENDED - use native substrate historical queries if possible.
    """

    def __init__(self):
        self._cache = {}  # {block_hash: {hotkey: commitment}}
        logger.warning(
            "Using HistoricalStateCache fallback - "
            "native substrate historical queries not available"
        )

    def store_commitment(self, block_hash: str, hotkey: str, commitment: dict):
        """Store commitment at specific block"""
        if block_hash not in self._cache:
            self._cache[block_hash] = {}
        self._cache[block_hash][hotkey] = commitment

    def get_commitment(self, block_hash: str, hotkey: str) -> dict | None:
        """Retrieve commitment at specific block"""
        return self._cache.get(block_hash, {}).get(hotkey)

    def store_all_commitments_at_block(
        self, block_hash: str, commitments: dict[str, dict]
    ):
        """Store all commitments at specific block"""
        self._cache[block_hash] = commitments.copy()

    def prune_old_blocks(self, keep_blocks: int = 1000):
        """Remove cached data for old blocks (keep memory bounded)"""
        if len(self._cache) > keep_blocks:
            # Keep only the N most recent blocks
            # (assumes block hashes are ordered by time, which is not always true)
            # Better: use block numbers as keys instead of hashes
            sorted_blocks = sorted(self._cache.keys())
            for block_hash in sorted_blocks[:-keep_blocks]:
                del self._cache[block_hash]
            logger.info(f"Pruned historical cache, kept {keep_blocks} blocks")


# Global cache instance (only used if substrate doesn't support historical reads)
_historical_cache = None


def get_historical_cache() -> HistoricalStateCache:
    """Get or create global historical state cache"""
    global _historical_cache
    if _historical_cache is None:
        _historical_cache = HistoricalStateCache()
    return _historical_cache
