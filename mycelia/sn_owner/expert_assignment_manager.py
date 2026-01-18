"""
SN Owner Expert Assignment Manager

Centralized control of which experts each miner is allowed to train.
Prevents miners from self-selecting experts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from mycelia.shared.app_logging import structlog
from mycelia.shared.config import OwnerConfig

logger = structlog.get_logger(__name__)

# Type aliases
ExpertMapping = Tuple[int, int]  # (my_expert_idx, org_expert_idx)
LayerAssignments = Dict[int, List[ExpertMapping]]  # layer_id -> list of mappings
MinerAssignment = Dict[int, LayerAssignments]  # expert_group_id -> layer assignments


class ExpertAssignmentManager:
    """
    Manages expert-to-miner assignments for the subnet.
    
    The SN owner uses this to:
    1. Define which experts each miner can train
    2. Serve authoritative assignments via API
    3. Track assignment history for auditing
    """

    def __init__(self, config: OwnerConfig):
        self.config = config
        self.assignment_path = Path(config.owner.assignment_storage_path if hasattr(config.owner, 'assignment_storage_path') else "data/expert_assignments")
        self.assignment_path.mkdir(parents=True, exist_ok=True)
        
        # Cache: {miner_hotkey: {expert_group_id: layer_assignments}}
        self.assignments: Dict[str, MinerAssignment] = {}
        self._load_assignments()

    def _load_assignments(self):
        """Load existing assignments from disk."""
        assignment_file = self.assignment_path / "assignments.json"
        if assignment_file.exists():
            try:
                with open(assignment_file, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to ints where needed
                    self.assignments = self._deserialize_assignments(data)
                logger.info("Loaded existing expert assignments", num_miners=len(self.assignments))
            except Exception as e:
                logger.error("Failed to load assignments", error=str(e))
                self.assignments = {}
        else:
            logger.info("No existing assignments found, starting fresh")

    def _save_assignments(self):
        """Save assignments to disk."""
        assignment_file = self.assignment_path / "assignments.json"
        try:
            with open(assignment_file, 'w') as f:
                json.dump(self._serialize_assignments(self.assignments), f, indent=2)
            logger.info("Saved expert assignments", num_miners=len(self.assignments))
        except Exception as e:
            logger.error("Failed to save assignments", error=str(e))

    def _serialize_assignments(self, assignments: Dict[str, MinerAssignment]) -> dict:
        """Convert to JSON-serializable format."""
        return {
            miner_hotkey: {
                str(group_id): {
                    str(layer_id): mappings  # tuples are lists in JSON
                    for layer_id, mappings in layers.items()
                }
                for group_id, layers in groups.items()
            }
            for miner_hotkey, groups in assignments.items()
        }

    def _deserialize_assignments(self, data: dict) -> Dict[str, MinerAssignment]:
        """Convert from JSON format back to proper types."""
        return {
            miner_hotkey: {
                int(group_id): {
                    int(layer_id): [tuple(m) for m in mappings]
                    for layer_id, mappings in layers.items()
                }
                for group_id, layers in groups.items()
            }
            for miner_hotkey, groups in data.items()
        }

    def get_assignment(
        self, 
        miner_hotkey: str, 
        expert_group_id: int
    ) -> LayerAssignments | None:
        """
        Get the expert assignment for a specific miner and expert group.
        
        Args:
            miner_hotkey: Miner's SS58 address
            expert_group_id: Expert group ID (0=math, 1=agentic, 2=planning, etc.)
            
        Returns:
            Layer assignments if authorized, None otherwise
        """
        if miner_hotkey not in self.assignments:
            logger.warning("Miner not found in assignments", miner=miner_hotkey)
            return None
            
        miner_groups = self.assignments[miner_hotkey]
        if expert_group_id not in miner_groups:
            logger.warning(
                "Miner not assigned to expert group",
                miner=miner_hotkey,
                group=expert_group_id,
            )
            return None
            
        return miner_groups[expert_group_id]

    def assign_expert_to_miner(
        self,
        miner_hotkey: str,
        expert_group_id: int,
        layer_assignments: LayerAssignments,
    ):
        """
        Assign specific experts to a miner.
        
        Args:
            miner_hotkey: Miner's SS58 address
            expert_group_id: Expert group ID
            layer_assignments: {layer_id: [(my_expert_id, org_expert_id), ...]}
        """
        if miner_hotkey not in self.assignments:
            self.assignments[miner_hotkey] = {}
            
        self.assignments[miner_hotkey][expert_group_id] = layer_assignments
        self._save_assignments()
        
        logger.info(
            "Assigned experts to miner",
            miner=miner_hotkey,
            group=expert_group_id,
            layers=len(layer_assignments),
        )

    def load_from_expert_group_configs(self, base_path: Path):
        """
        Load expert assignments from expert_groups/ config files.
        
        This creates a default "round-robin" assignment where experts
        are distributed evenly across registered miners.
        
        Args:
            base_path: Path to expert_groups/ directory
        """
        from mycelia.shared.config import TaskCfg
        
        task_folders = [d for d in base_path.iterdir() if d.is_dir()]
        
        # Load all expert group definitions
        expert_groups: Dict[int, LayerAssignments] = {}
        for task_folder in task_folders:
            config_file = task_folder / "config.yaml"
            assignment_file = task_folder / "expert_assignment.json"
            
            if not config_file.exists() or not assignment_file.exists():
                continue
                
            task_config = TaskCfg.from_path(config_file)
            
            with open(assignment_file, 'r') as f:
                raw_assignment = json.load(f)
                
            # Convert to proper format
            layer_assignments: LayerAssignments = {}
            for layer_id_str, pair_list in raw_assignment.items():
                layer_id = int(layer_id_str)
                mappings: List[ExpertMapping] = [tuple(pair) for pair in pair_list]
                layer_assignments[layer_id] = mappings
                
            expert_groups[task_config.expert_group_id] = layer_assignments
            
        logger.info("Loaded expert group definitions", num_groups=len(expert_groups))
        return expert_groups

    def auto_assign_miners(
        self,
        miner_hotkeys: List[str],
        base_path: Path,
        strategy: str = "round_robin",
    ):
        """
        Automatically assign experts to miners using a specified strategy.
        
        Args:
            miner_hotkeys: List of registered miner hotkeys
            base_path: Path to expert_groups/ directory
            strategy: Assignment strategy ("round_robin", "balanced", "random")
        """
        expert_groups = self.load_from_expert_group_configs(base_path)
        
        if strategy == "round_robin":
            self._assign_round_robin(miner_hotkeys, expert_groups)
        elif strategy == "balanced":
            self._assign_balanced(miner_hotkeys, expert_groups)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        logger.info(
            "Auto-assigned experts to miners",
            num_miners=len(miner_hotkeys),
            strategy=strategy,
        )

    def _assign_round_robin(
        self,
        miners: List[str],
        expert_groups: Dict[int, LayerAssignments],
    ):
        """
        Distribute experts evenly across miners in round-robin fashion.
        
        Each expert group's experts are split among all miners.
        """
        if not miners:
            logger.warning("No miners provided for assignment")
            return
            
        num_miners = len(miners)
        
        for group_id, layer_assignments in expert_groups.items():
            # For each layer, distribute experts among miners
            for layer_id, expert_mappings in layer_assignments.items():
                for i, mapping in enumerate(expert_mappings):
                    miner_idx = i % num_miners
                    miner = miners[miner_idx]
                    
                    # Initialize nested dicts if needed
                    if miner not in self.assignments:
                        self.assignments[miner] = {}
                    if group_id not in self.assignments[miner]:
                        self.assignments[miner][group_id] = {}
                    if layer_id not in self.assignments[miner][group_id]:
                        self.assignments[miner][group_id][layer_id] = []
                    
                    # Assign this expert to this miner
                    self.assignments[miner][group_id][layer_id].append(mapping)
        
        self._save_assignments()

    def _assign_balanced(
        self,
        miners: List[str],
        expert_groups: Dict[int, LayerAssignments],
    ):
        """
        Distribute experts to balance total workload across miners.
        
        Attempts to give each miner approximately the same number of experts.
        """
        # Count total experts
        total_experts = sum(
            len(mappings)
            for layers in expert_groups.values()
            for mappings in layers.values()
        )
        
        experts_per_miner = total_experts // len(miners)
        logger.info(
            "Balanced assignment",
            total_experts=total_experts,
            miners=len(miners),
            target_per_miner=experts_per_miner,
        )
        
        # Simple balanced distribution (can be improved)
        self._assign_round_robin(miners, expert_groups)

    def get_all_assignments(self) -> Dict[str, MinerAssignment]:
        """Get all miner assignments for debugging/auditing."""
        return self.assignments.copy()

    def remove_miner(self, miner_hotkey: str):
        """Remove a miner's assignment (e.g., if they leave the network)."""
        if miner_hotkey in self.assignments:
            del self.assignments[miner_hotkey]
            self._save_assignments()
            logger.info("Removed miner assignment", miner=miner_hotkey)

