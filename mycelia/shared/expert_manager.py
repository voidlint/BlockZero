from __future__ import annotations

import json
import random
import re
import requests
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from mycelia.shared.app_logging import structlog
from mycelia.shared.config import TaskCfg, WorkerConfig

logger = structlog.getLogger(__name__)

# ------------------------------------------------------------
# Expert Manager
# ------------------------------------------------------------
ExpertMapping = tuple[int, int]  # (my_expert_idx, org_expert_idx)
LayerAssignments = dict[int, list[ExpertMapping]]  # layer_id -> list of mappings
ExpertAssignments = dict[int, LayerAssignments]  # group_id -> layer assignments


class ExpertManager:
    """
    Manages expert-aware grouping for distributed Mixture-of-Experts training.

    Responsibilities
    ----------------
    * Discover which transformer layers contain experts (by scanning state_dict keys).
    * Split ranks into `num_worker_groups` groups.
    * Split experts (per expert layer) into the same number of groups.

    Attributes
    ----------
    expert_layers : list[int]
        Indices of layers that contain experts (best-effort heuristic).
    rank_group_assignment : Dict[int, list[int]]
        Mapping of group_id -> ranks in that group.
    expert_group_assignment : Dict[int, Dict[int, list[int]]]
        Mapping of layer -> (group_id -> expert IDs in that group).
    """

    def __init__(self, config: WorkerConfig, model: nn.Module | None = None):
        if model is not None:
            self.set_expert_layers(model)
        else:
            self.expert_layers = None

        self.expert_group_assignment = self.load_expert_group_assignment(config)
        self.validate_unique_mycelia_expert_ids()
        self.validate_expert_layers()

    @property
    def num_expert_groups(self) -> int:
        """
        Number of expert groups present in the assignment.
        Directly equal to number of group_id keys.
        """
        return len(self.expert_group_assignment)

    @property
    def num_experts(self) -> int:
        """
        Total count of unique my_expert_idx across all groups/layers.
        """
        expert_ids = set()

        for layers in self.expert_group_assignment.values():
            for mappings in layers.values():
                for my_expert_idx, _ in mappings:
                    expert_ids.add(my_expert_idx)

        return len(expert_ids)

    def get_num_experts_in_group(self, group_id) -> int:
        """
        Total count of unique my_expert_idx across all groups/layers.
        """
        expert_ids = set()

        for _, mappings in self.expert_group_assignment[group_id].items():
            for my_expert_idx, _ in mappings:
                expert_ids.add(my_expert_idx)

        return len(expert_ids)

    def set_expert_layers(self, model: nn.Module) -> list[int]:
        """
        Inspect model state_dict keys to locate layers that include experts.

        Heuristic: looks for keys like "{layer_idx}.mlp.experts..." (tweak for your arch).
        """
        sd_keys = model.state_dict().keys()
        num_layers = getattr(getattr(model, "config", object()), "num_hidden_layers", None)
        if num_layers is None:
            logger.warning("Model has no config.num_hidden_layers; scanning keys without layer bounds.")

        expert_layers: list[int] = []
        # If num_layers is unknown, fall back to a generous range (0..255).
        layer_range = range(num_layers if isinstance(num_layers, int) else 256)
        for layer_id in layer_range:
            pattern = f"{layer_id}.mlp.experts"
            if any(pattern in k for k in sd_keys):
                expert_layers.append(layer_id)

        if not expert_layers:
            logger.info("No expert-bearing layers found by heuristic; check naming or discovery logic.")
        else:
            logger.info(f"Detected expert layerss: {expert_layers}")

        self.expert_layers = expert_layers
        self.validate_expert_layers()

    # ---- loading ----
    def load_expert_group_assignment(self, config) -> ExpertAssignments:
        """
        Load expert assignments.
        
        For MINERS: Fetch from SN owner API (enforces centralized control)
        For VALIDATORS/OWNER: Load from local files
        """
        # Check if this is a miner that should fetch from API
        role = getattr(config, 'role', None)
        
        if role == "miner":
            # Try to fetch from API, but fall back to local files if wallet not configured
            try:
                if hasattr(config, 'wallet') and config.wallet is not None:
                    return self._fetch_assignment_from_sn_owner(config)
                else:
                    logger.warning("Wallet not configured, falling back to local expert assignment files")
                    return self._load_assignment_from_local_files(config)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Failed to fetch from API, falling back to local files: {e}")
                return self._load_assignment_from_local_files(config)
        else:
            return self._load_assignment_from_local_files(config)

    def _fetch_assignment_from_sn_owner(self, config) -> ExpertAssignments:
        """
        Fetch expert assignment from SN owner API.
        
        This enforces centralized control - miners cannot choose their own experts.
        """
        try:
            # Get miner hotkey
            if not hasattr(config, 'wallet') or config.wallet is None:
                raise ValueError("Miner must have wallet configured to fetch assignments")
            
            miner_hotkey = config.wallet.hotkey.ss58_address
            expert_group_id = config.task.expert_group_id
            
            # Fetch from SN owner API
            owner_url = getattr(config, 'owner_url', 'http://localhost:7000')
            response = requests.get(
                f"{owner_url}/get-expert-assignment",
                params={
                    "miner_hotkey": miner_hotkey,
                    "expert_group_id": expert_group_id,
                },
                timeout=30,
            )
            
            if response.status_code == 403:
                raise PermissionError(
                    f"Miner {miner_hotkey} not authorized for expert group {expert_group_id}.\n"
                    "Contact subnet owner for assignment."
                )
            
            response.raise_for_status()
            assignment_data = response.json()
            
            # ✅ SECURITY: Verify signature from SN owner
            if "signature" not in assignment_data:
                raise ValueError("Assignment missing signature from SN owner")
            
            sn_owner_hotkey = assignment_data.get("sn_owner_hotkey")
            if not sn_owner_hotkey:
                raise ValueError("Assignment missing sn_owner_hotkey")
            
            # Verify signature using signature_utils
            from mycelia.shared.signature_utils import verify_assignment_signature
            
            is_valid = verify_assignment_signature(assignment_data, sn_owner_hotkey)
            if not is_valid:
                raise PermissionError(
                    f"Invalid signature on expert assignment from {sn_owner_hotkey[:16]}..."
                )
            
            logger.info(
                "✓ Verified signature on expert assignment",
                sn_owner=sn_owner_hotkey[:16] + "...",
            )
            
            # Convert from API format to internal format
            layer_assignments: LayerAssignments = {}
            for layer_id_str, mappings_list in assignment_data["layer_assignments"].items():
                layer_id = int(layer_id_str)
                mappings: list[ExpertMapping] = [tuple(m) for m in mappings_list]
                layer_assignments[layer_id] = mappings
            
            logger.info(
                "✓ Fetched expert assignment from SN owner",
                miner=miner_hotkey[:8] + "...",
                expert_group_id=expert_group_id,
                num_layers=len(layer_assignments),
                owner_url=owner_url,
            )
            
            return {expert_group_id: layer_assignments}
            
        except requests.exceptions.RequestException as e:
            logger.error(
                "Failed to fetch expert assignment from SN owner",
                error=str(e),
                owner_url=getattr(config, 'owner_url', 'http://localhost:7000'),
            )
            raise RuntimeError(
                f"Cannot start miner: Failed to fetch expert assignment from SN owner.\n"
                f"Error: {e}\n"
                f"Make sure the SN owner service is running and accessible."
            )
        except PermissionError:
            raise
        except Exception as e:
            logger.error("Unexpected error fetching assignment", error=str(e))
            raise

    def _load_assignment_from_local_files(self, config) -> ExpertAssignments:
        """
        Load expert assignments from local expert_groups/ config files.
        
        Used by validators and SN owner.
        """
        base_path: Path = config.task.base_path
        task_folders = [d for d in base_path.iterdir() if d.is_dir()]

        expert_assignments: ExpertAssignments = {}

        for task_folder in task_folders:
            config_path = task_folder / "config.yaml"
            assignment_path = task_folder / "expert_assignment.json"
            
            # Skip folders without both config.yaml and expert_assignment.json
            if not config_path.exists():
                logger.debug("Skipping folder without config.yaml", task_folder=task_folder.name)
                continue
            if not assignment_path.exists():
                logger.debug("Skipping folder without expert_assignment.json", task_folder=task_folder.name)
                continue
            
            logger.debug("loading expert group assignment from folder", task_folder)
            # Load per-task config (to get expert_group_id)
            task_config = TaskCfg.from_path(config_path)

            # Load raw JSON assignment
            with open(assignment_path, encoding="utf-8") as f:
                raw_assignment = json.load(f)

            # raw_assignment: Dict[str, list[list[int]]]
            # Convert to: LayerAssignments (Dict[int, list[tuple[int, int]]])
            layer_assignments: LayerAssignments = {}
            for layer_id_str, pair_list in raw_assignment.items():
                layer_id = int(layer_id_str)
                # Ensure we store tuples of ints, not lists
                mappings: list[ExpertMapping] = [tuple(pair) for pair in pair_list]
                layer_assignments[layer_id] = mappings

            # Map this task's expert_group_id -> its layer assignments
            expert_assignments[task_config.expert_group_id] = layer_assignments

        logger.info(
            "Loaded expert assignments from local files",
            num_groups=len(expert_assignments),
        )

        return expert_assignments

    # ---- Check correctness ----
    def validate_unique_mycelia_expert_ids(self) -> None:
        """
        Check that `my_expert_idx` is unique within each (group_id, layer_id).

        In other words, for any fixed (group_id, layer_id) pair, the same
        `my_expert_idx` must not appear more than once in its mapping list.
        Reuse of a `my_expert_idx` across *different* groups or layers is allowed.

        Raises
        ------
        ValueError
            If any `my_expert_idx` appears more than once in the same
            (group_id, layer_id).
        """
        duplicates: list[str] = []

        for group_id, layers in self.expert_group_assignment.items():
            for layer_id, mappings in layers.items():
                seen_in_layer: set[int] = set()
                for my_expert_idx, _ in mappings:
                    if my_expert_idx in seen_in_layer:
                        duplicates.append(
                            f"mycelia_expert_idx={my_expert_idx} duplicated "
                            f"within (group={group_id}, layer={layer_id})"
                        )
                    else:
                        seen_in_layer.add(my_expert_idx)

        if duplicates:
            # Show a concise but useful error
            msg = "Duplicate mycelia_expert_idx found within expert groups/layers:\n" + "\n".join(duplicates[:10])
            if len(duplicates) > 10:
                msg += f"\n... and {len(duplicates) - 10} more"
            raise ValueError(msg)

    def validate_expert_layers(self) -> None:
        """
        Validate that for every expert group, the set of layer_ids matches
        exactly the required set in expert_layers (inclusive + exclusive).

        Raises
        ------
        ValueError
            If any group is missing layers or has extra layers.
        """
        if self.expert_layers is None:
            return

        expected = set(self.expert_layers)
        errors = []

        for group_id, layers in self.expert_group_assignment.items():
            actual = set(layers.keys())

            missing = expected - actual
            extra = actual - expected

            if missing or extra:
                msg = [f"Group {group_id} layer mismatch:"]

                if missing:
                    msg.append(f"  Missing layers: {sorted(missing)}")
                if extra:
                    msg.append(f"  Extra layers: {sorted(extra)}")

                errors.append("\n".join(msg))

        if errors:
            raise ValueError("Expert layer validation failed:\n\n" + "\n\n".join(errors))


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def is_expert_param(name: str) -> bool:
    """Heuristic to detect MoE expert parameters by name."""
    return "expert" in name  # customize if needed (e.g., "experts.")


def get_layer_expert_id(layer_name: str) -> tuple[int | None, int | None]:
    """
    Extract (layer_id, expert_id) from a parameter name.

    Examples
    --------
    "model.layers.3.mlp.experts.7.w1.weight" -> (3, 7)
    "model.layers.5.mlp.gate.weight"         -> (5, None)
    """
    m = re.search(r"layers\.(\d+)(?:\.mlp\.experts\.(\d+))?", layer_name)
    if not m:
        return None, None
    layer_id = int(m.group(1))
    expert_id = int(m.group(2)) if m.group(2) is not None else None
    return layer_id, expert_id


def split_into_groups(
    lst: list[int], num_groups: int, shuffle: bool = False, seed: int | None = 123
) -> dict[int, list[int]]:
    """
    Deterministically split a list of items into `num_groups` interleaved buckets.

    Parameters
    ----------
    lst : list[int]
        Items to split (e.g., ranks or expert IDs).
    num_groups : int
        Number of buckets to produce.
    seed : Optional[int]
        Seed for reproducible shuffling. If None, keeps original order.

    Returns
    -------
    Dict[int, list[int]]
        Mapping: group_id -> sublist of items.

    Notes
    -----
    Uses a local RNG so global randomness is unaffected.
    """
    if num_groups <= 0:
        raise ValueError("num_groups must be >= 1")

    if shuffle:
        shuffled = lst[:]
        if seed is not None:
            rnd = random.Random(seed)
            rnd.shuffle(shuffled)

        return {i: shuffled[i::num_groups] for i in range(num_groups)}

    else:
        return {i: lst[i * (len(lst) // num_groups) : (i + 1) * (len(lst) // num_groups)] for i in range(num_groups)}


def create_expert_groups(
    my_rank: int, rank_group_assignment: Mapping[int, Iterable[int]]
) -> tuple[int, dict[int, dist.ProcessGroup]]:
    """
    Create torch.distributed process groups for each expert group.

    Parameters
    ----------
    my_rank : int
        This process' global rank.
    rank_group_assignment : Mapping[int, Iterable[int]]
        Mapping of group_id -> ranks in that group.

    Returns
    -------
    tuple[int, Dict[int, ProcessGroup]]
        (group_ids, groups_by_id)

    Notes
    -----
    * Requires `dist.is_initialized()` to be True.
    * Each call will create new groups; reuse the returned dict across calls in your job.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before creating groups")

    expert_groups: dict[int, dist.ProcessGroup] = {}
    group_ids: int | None = None

    for group_id, ranks in rank_group_assignment.items():
        group = dist.new_group(ranks=ranks)
        expert_groups[group_id] = group
        if my_rank in ranks:
            group_ids = group_id

    if group_ids is None:
        raise ValueError(f"Rank {my_rank} not present in any provided group assignment")

    return group_ids, expert_groups


# ------------------------------------------------------------
# Synchronization primitives
# ------------------------------------------------------------
def _named_params(model: nn.Module) -> dict[str, nn.Parameter]:
    """Return a dict name -> parameter for stable matching across models."""
    return dict(model.named_parameters())


def populate_global_grads_from_local(
    global_model: nn.Module, model: nn.Module, shared_only: bool = False, weight: float = 0.2
) -> None:
    """
    Average the differences for *shared* (non-expert) parameters across all ranks.

    Workflow
    --------
    * For each shared param `p` in `model` and corresponding `g` in `global_model`:
        grad_g = (g.data - p.data)
        all_reduce(grad_g, AVG)  # average difference across workers
      (You typically apply these "gradients" via an optimizer on `global_model` later.)

    Notes
    -----
    * We avoid relying on parameter iteration order by matching by name.
    * Uses `.data` to avoid autograd tracking (intentional, as these are sync ops).
    """
    local_named = _named_params(model)
    global_named = _named_params(global_model)

    for name, p in local_named.items():
        if shared_only and is_expert_param(name):
            continue

        g = global_named.get(name)
        if g is None:
            logger.warning(f"Shared param '{name}' not found in global model; skipping.")
            continue

        diff = g.data - p.data
        if g.grad is None:
            g.grad = diff * weight
        else:
            g.grad += diff * weight


def sync_weights(rank: int, global_model: nn.Module, shared_only: bool = False) -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before sync")

    global_named = _named_params(global_model)

    for name, g in global_named.items():
        if shared_only and is_expert_param(name):
            continue

        dist.all_reduce(g.grad, op=dist.ReduceOp.AVG)


def sync_expert_weights(
    rank: int,
    global_model: nn.Module,
    model: nn.Module,
    group_ids: int,
    expert_groups: Mapping[int, dist.ProcessGroup],
) -> None:
    """
    Average the differences for *expert* parameters within this expert group only.

    Parameters
    ----------
    group_ids : int
        ID of the group this rank belongs to.
    expert_groups : Mapping[int, ProcessGroup]
        Mapping from group ID to its ProcessGroup.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before sync")

    group = expert_groups.get(group_ids)
    if group is None:
        raise KeyError(f"No process group for group_ids={group_ids}")

    local_named = _named_params(model)
    global_named = _named_params(global_model)

    for name, p in local_named.items():
        if not is_expert_param(name):
            continue
        g = global_named.get(name)
        if g is None:
            logger.warning(f"[rank {rank}] Expert param '{name}' not found in global model; skipping.")
            continue

        diff = g.data - p.data
        g.grad = diff
        dist.all_reduce(g.grad, op=dist.ReduceOp.AVG, group=group)


def broadcast_weights(
    model: nn.Module,
    group_ids: int,
    rank_group_assignment: Mapping[int, Iterable[int]],
    expert_groups: Mapping[int, dist.ProcessGroup],
) -> None:
    """
    Broadcast parameters so every rank has a consistent view.

    Behavior
    --------
    * Expert params: broadcast **within** group from the lowest-rank member.
    * Shared params: broadcast **globally** from rank 0.

    Notes
    -----
    Ensure collectives are called by all ranks consistently.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before broadcast")

    src_expert_rank = min(rank_group_assignment[group_ids])
    expert_group = expert_groups[group_ids]

    for name, p in model.named_parameters():
        if is_expert_param(name):
            dist.broadcast(p.data, src=src_expert_rank, group=expert_group)
        else:
            dist.broadcast(p.data, src=0)


def get_weight_sum(model: nn.Module, shared: bool = True) -> tuple[str, torch.Tensor]:
    """
    Return the (name, sum) for the first parameter that matches the filter.

    Parameters
    ----------
    model : nn.Module
        Model to inspect.
    shared : bool
        If True, look at non-expert (shared) params; else look at expert params.

    Returns
    -------
    Optional[tuple[str, Tensor]]
        (parameter_name, tensor_sum) for the first matching parameter, or None if none match.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if shared and not is_expert_param(name):
                return name, p.data.sum()
            if not shared and is_expert_param(name):
                return name, p.data.sum()
