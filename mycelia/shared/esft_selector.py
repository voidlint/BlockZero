"""
Expert-Specialized Fine-Tuning (ESFT) Expert Selection

Based on "Let the Expert Stick to His Last" (2407.01906v2)

Key findings from paper:
1. Task-specific expert specialization: each task uses a concentrated subset of experts
2. Cross-task differentiation: different tasks use different expert subsets
3. Fine-grained segmentation enables better specialization
4. Training only task-relevant experts matches FFT while preserving general ability

Implementation:
- Sample small subset of task data (32 samples × 4096 seq len)
- Compute expert relevance scores (Gate or Token method)
- Select experts via cumulative threshold p
- Train only selected experts (freeze others + shared expert)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass
from pathlib import Path
import json

from mycelia.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ESFTConfig:
    """ESFT configuration."""

    # Expert selection method
    method: Literal["gate", "token"] = "token"  # Paper: Token is more stable

    # Cumulative threshold for expert selection
    # Paper: p=0.1 for gate, p=0.2 for token
    threshold: float = 0.2

    # Data sampling for expert selection
    num_samples: int = 32  # Paper: 32 samples is robust
    sequence_length: int = 4096

    # What to train (based on paper Table 3)
    train_selected_experts: bool = True
    train_shared_expert: bool = False  # Paper: degrades general ability
    train_gates: bool = False  # Paper: minimal benefit
    train_other: bool = False  # Attention, embeddings

    # Save expert selection results
    save_selection: bool = True
    selection_path: Path | None = None


class ESFTExpertSelector:
    """
    Select task-relevant experts using ESFT methodology.

    Usage:
        selector = ESFTExpertSelector(model, config)

        # 1. Profile expert usage on task data
        for batch in sample_dataloader:
            selector.accumulate_expert_stats(batch)

        # 2. Select experts per layer
        selected_experts = selector.select_experts()

        # 3. Freeze non-selected experts
        selector.freeze_non_selected_experts(selected_experts)
    """

    def __init__(
        self,
        model: nn.Module,
        config: ESFTConfig,
        num_experts: int = 128,
        shared_expert_id: int = 127,
    ):
        self.model = model
        self.config = config
        self.num_experts = num_experts
        self.shared_expert_id = shared_expert_id

        # Statistics accumulation
        # expert_gate_scores[layer_idx][expert_idx] = sum of gate values
        # expert_token_counts[layer_idx][expert_idx] = count of tokens selecting this expert
        self.expert_gate_scores: Dict[int, torch.Tensor] = {}
        self.expert_token_counts: Dict[int, torch.Tensor] = {}
        self.total_tokens: int = 0

        self._reset_stats()

    def _reset_stats(self):
        """Reset accumulated statistics."""
        self.expert_gate_scores = {}
        self.expert_token_counts = {}
        self.total_tokens = 0
        logger.info("Reset ESFT expert statistics")

    def _get_moe_layers(self) -> List[Tuple[int, nn.Module]]:
        """
        Get all MoE layers from model.

        Returns:
            List of (layer_idx, moe_block) tuples
        """
        moe_layers = []

        # For Qwen3-VL-MoE: model.language_model.layers[i].mlp
        if hasattr(self.model, 'language_model'):
            layers = self.model.language_model.layers
        elif hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            raise ValueError("Cannot find model layers")

        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                moe_layers.append((layer_idx, layer.mlp))

        logger.info(f"Found {len(moe_layers)} MoE layers")
        return moe_layers

    @torch.no_grad()
    def accumulate_expert_stats(
        self,
        batch: Dict[str, torch.Tensor],
        record_routing: bool = True,
    ):
        """
        Accumulate expert usage statistics on a batch.

        Args:
            batch: Input batch (input_ids, attention_mask, labels, etc.)
            record_routing: Whether to record routing information

        Note:
            This requires the model to output router_logits.
            Set model.config.output_router_logits = True
        """
        # Forward pass to get routing information
        outputs = self.model(**batch, output_router_logits=True)

        if not hasattr(outputs, 'router_logits') or outputs.router_logits is None:
            raise ValueError(
                "Model must output router_logits. "
                "Set model.config.output_router_logits = True"
            )

        # router_logits: tuple of (batch_size * seq_len, num_experts) per layer
        router_logits = outputs.router_logits

        batch_tokens = batch['input_ids'].shape[0] * batch['input_ids'].shape[1]
        self.total_tokens += batch_tokens

        for layer_idx, logits in enumerate(router_logits):
            if layer_idx not in self.expert_gate_scores:
                self.expert_gate_scores[layer_idx] = torch.zeros(
                    self.num_experts,
                    device=logits.device,
                    dtype=torch.float32
                )
                self.expert_token_counts[layer_idx] = torch.zeros(
                    self.num_experts,
                    device=logits.device,
                    dtype=torch.long
                )

            # Compute gate values (softmax of logits)
            gate_values = torch.softmax(logits, dim=-1)  # (batch*seq, num_experts)

            # Method 1: Average Gate Score (ESFT-Gate)
            # Sum gate values for each expert
            self.expert_gate_scores[layer_idx] += gate_values.sum(dim=0)

            # Method 2: Token Selection Ratio (ESFT-Token)
            # Count tokens that selected each expert (top-K)
            topk = self.model.config.text_config.num_experts_per_tok if hasattr(self.model.config, 'text_config') else 8
            _, selected_experts = torch.topk(gate_values, topk, dim=-1)

            # One-hot encode selections and sum
            expert_selected = torch.zeros_like(gate_values)
            expert_selected.scatter_(1, selected_experts, 1.0)
            self.expert_token_counts[layer_idx] += expert_selected.sum(dim=0).long()

    def compute_expert_relevance(
        self,
        layer_idx: int,
        method: Literal["gate", "token"] = "token",
    ) -> torch.Tensor:
        """
        Compute expert relevance scores for a layer.

        Args:
            layer_idx: Layer index
            method: "gate" (average gate score) or "token" (selection ratio)

        Returns:
            relevance_scores: (num_experts,) tensor of relevance scores

        Paper formulas:
            ESFT-Gate: R_i^l = (1/T) * Σ_t g_i^l(x_t)
            ESFT-Token: R_i^l = (1/(T*K)) * Σ_t 1[expert_i selected for token_t]
        """
        if method == "gate":
            # Average gate score
            relevance = self.expert_gate_scores[layer_idx] / self.total_tokens
        else:  # token
            # Token selection ratio (normalized by K experts per token)
            topk = 8  # Qwen3-VL-MoE uses top-8
            relevance = self.expert_token_counts[layer_idx] / (self.total_tokens * topk)

        return relevance

    def select_experts(
        self,
        method: str | None = None,
        threshold: float | None = None,
    ) -> Dict[int, List[int]]:
        """
        Select task-relevant experts per layer using cumulative threshold.

        Args:
            method: Override config method ("gate" or "token")
            threshold: Override config threshold (p ∈ (0, 1])

        Returns:
            selected_experts: {layer_idx: [expert_ids]}

        Paper formula:
            E_s^l = {smallest set of experts where Σ R_i^l >= p}
        """
        method = method or self.config.method
        threshold = threshold or self.config.threshold

        selected_experts = {}
        total_selected = 0

        for layer_idx in sorted(self.expert_gate_scores.keys()):
            # Compute relevance scores
            relevance = self.compute_expert_relevance(layer_idx, method=method)

            # Sort experts by relevance (descending)
            sorted_indices = torch.argsort(relevance, descending=True)
            sorted_relevance = relevance[sorted_indices]

            # Cumulative sum
            cumsum = torch.cumsum(sorted_relevance, dim=0)

            # Find smallest set where cumsum >= threshold
            # Paper: "smallest set E_s^l such that Σ R_i^l >= p"
            num_selected = (cumsum < threshold).sum().item() + 1
            num_selected = min(num_selected, self.num_experts)

            # Select top experts
            selected = sorted_indices[:num_selected].tolist()

            # Always exclude shared expert from trainable set
            # Paper Table 3: training shared expert degrades general ability
            if self.shared_expert_id in selected:
                selected.remove(self.shared_expert_id)

            selected_experts[layer_idx] = selected
            total_selected += len(selected)

            logger.info(
                f"Layer {layer_idx}: selected {len(selected)}/{self.num_experts} experts "
                f"(threshold={threshold:.2f}, method={method})"
            )

        avg_selected = total_selected / len(selected_experts) if selected_experts else 0
        logger.info(
            f"ESFT selection complete: avg {avg_selected:.1f} experts/layer "
            f"({avg_selected/self.num_experts*100:.1f}% of total)"
        )

        # Save selection results
        if self.config.save_selection and self.config.selection_path:
            self._save_selection(selected_experts, method, threshold)

        return selected_experts

    def _save_selection(
        self,
        selected_experts: Dict[int, List[int]],
        method: str,
        threshold: float,
    ):
        """Save expert selection results to JSON."""
        save_path = self.config.selection_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute statistics per layer
        stats = {}
        for layer_idx, experts in selected_experts.items():
            relevance = self.compute_expert_relevance(layer_idx, method=method)

            stats[f"layer_{layer_idx}"] = {
                "selected_experts": experts,
                "num_selected": len(experts),
                "selection_ratio": len(experts) / self.num_experts,
                "relevance_scores": relevance.tolist(),
            }

        result = {
            "method": method,
            "threshold": threshold,
            "total_tokens": self.total_tokens,
            "num_experts": self.num_experts,
            "shared_expert_id": self.shared_expert_id,
            "avg_experts_per_layer": sum(len(e) for e in selected_experts.values()) / len(selected_experts),
            "layer_stats": stats,
        }

        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved ESFT expert selection to {save_path}")

    def freeze_non_selected_experts(
        self,
        selected_experts: Dict[int, List[int]],
    ):
        """
        Freeze all parameters except selected experts.

        Based on paper Table 3: train only task-relevant non-shared experts.

        Args:
            selected_experts: {layer_idx: [expert_ids]} from select_experts()
        """
        moe_layers = self._get_moe_layers()

        # First: freeze ALL parameters
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_params = 0
        total_params = 0

        for layer_idx, moe_block in moe_layers:
            if layer_idx not in selected_experts:
                continue

            selected = selected_experts[layer_idx]

            # Unfreeze only selected experts
            if hasattr(moe_block, 'experts'):
                for expert_id in selected:
                    expert_key = str(expert_id)
                    if expert_key in moe_block.experts:
                        expert = moe_block.experts[expert_key]
                        for param in expert.parameters():
                            param.requires_grad = True
                            trainable_params += param.numel()

            # Count total expert params
            if hasattr(moe_block, 'experts'):
                for expert in moe_block.experts.values():
                    for param in expert.parameters():
                        total_params += param.numel()

        reduction = 1 - (trainable_params / total_params) if total_params > 0 else 0

        logger.info(
            f"ESFT parameter freezing complete:\n"
            f"  Trainable: {trainable_params:,} ({trainable_params/1e9:.2f}B)\n"
            f"  Total: {total_params:,} ({total_params/1e9:.2f}B)\n"
            f"  Reduction: {reduction*100:.1f}%"
        )


def run_esft_expert_selection(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: ESFTConfig,
    num_experts: int = 128,
    shared_expert_id: int = 127,
    allow_miner_execution: bool = False,
) -> Dict[int, List[int]]:
    """
    Run ESFT expert selection pipeline.

    **IMPORTANT FOR DISTRIBUTED TRAINING**:
    In a subnet with multiple miners, the subnet OWNER should run this ONCE
    and publish the expert assignment to all miners. Miners then load the
    published assignment instead of running selection independently.

    **SECURITY**: This function is RESTRICTED to SN owner only. Miners attempting
    to run this will be detected and penalized by validators.

    Workflow:
        1. Subnet owner: Run ESFT on representative data → save selection
        2. Miners: Load published selection → train same experts

    Args:
        model: MoE model to profile
        dataloader: Small sample of task data (32 samples recommended)
        config: ESFT configuration
        num_experts: Total number of experts
        shared_expert_id: Shared expert ID to exclude from training
        allow_miner_execution: INTERNAL USE ONLY - Set to True to bypass security check
                               (used for testing only)

    Returns:
        selected_experts: {layer_idx: [expert_ids]} to train

    Raises:
        PermissionError: If called from miner context without override flag

    Example (Subnet Owner):
        >>> # Subnet owner runs ESFT once per task
        >>> config = ESFTConfig(
        ...     method="token",
        ...     threshold=0.2,
        ...     save_selection=True,
        ...     selection_path=Path("configs/esft_math_experts.json"),
        ... )
        >>> selected = run_esft_expert_selection(model, sample_dataloader, config)
        >>>
        >>> # Publish configs/esft_math_experts.json to all miners

    Example (Miner):
        >>> # Miner loads subnet owner's published assignment
        >>> selected = load_expert_assignment("configs/esft_math_experts.json")
        >>> apply_expert_assignment(model, selected)
        >>> # Now train with same experts as all other miners
    """
    # SECURITY CHECK: Prevent miners from running ESFT locally
    import os
    import sys
    
    # Check if we're in a miner context
    is_miner_context = False
    
    # Method 1: Check if miner modules are in the call stack
    frame = sys._getframe()
    while frame:
        frame_filename = frame.f_code.co_filename
        if 'mycelia/miner/' in frame_filename and 'train.py' in frame_filename:
            is_miner_context = True
            break
        frame = frame.f_back
    
    # Method 2: Check environment variable (set by miner startup)
    if os.environ.get('MYCELIA_ROLE') == 'miner':
        is_miner_context = True
    
    if is_miner_context and not allow_miner_execution:
        error_msg = (
            "SECURITY ERROR: ESFT expert selection cannot be run by miners!\n\n"
            "Miners MUST use expert assignments provided by the subnet owner.\n"
            "Running ESFT locally is considered a protocol violation and will result in:\n"
            "  - Submission rejection by validators\n"
            "  - Zero rewards\n"
            "  - Potential blacklisting\n\n"
            "To train as a miner:\n"
            "  1. Fetch assignment from SN owner (done automatically)\n"
            "  2. Train only the assigned experts\n\n"
            "If you are the subnet owner and seeing this error, set:\n"
            "  allow_miner_execution=True (for testing only)\n"
            "  or run from an sn_owner context"
        )
        logger.error(
            "Unauthorized ESFT execution attempt detected",
            context="miner",
            caller=sys._getframe(1).f_code.co_filename,
        )
        raise PermissionError(error_msg)
    
    selector = ESFTExpertSelector(model, config, num_experts, shared_expert_id)

    # Enable router logits output
    original_output_router_logits = getattr(
        model.config.text_config if hasattr(model.config, 'text_config') else model.config,
        'output_router_logits',
        False
    )
    if hasattr(model.config, 'text_config'):
        model.config.text_config.output_router_logits = True
    else:
        model.config.output_router_logits = True

    try:
        # Profile expert usage
        model.eval()
        logger.info(f"Profiling expert usage on {config.num_samples} samples...")

        for i, batch in enumerate(dataloader):
            if i >= config.num_samples:
                break

            # Move batch to model device
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            selector.accumulate_expert_stats(batch)

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{config.num_samples} samples")

        # Select experts
        logger.info("Selecting task-relevant experts...")
        selected_experts = selector.select_experts()

        # Freeze non-selected
        logger.info("Freezing non-selected parameters...")
        selector.freeze_non_selected_experts(selected_experts)

        return selected_experts

    finally:
        # Restore original setting
        if hasattr(model.config, 'text_config'):
            model.config.text_config.output_router_logits = original_output_router_logits
        else:
            model.config.output_router_logits = original_output_router_logits


# ============================================================================
# Distributed Training: Load Published Expert Assignments
# ============================================================================

def load_expert_assignment(selection_path: str | Path) -> Dict[int, List[int]]:
    """
    Load expert assignment published by subnet owner.

    Args:
        selection_path: Path to ESFT selection JSON file

    Returns:
        selected_experts: {layer_idx: [expert_ids]}

    Example (Miner):
        >>> # Load subnet owner's published assignment
        >>> selected = load_expert_assignment("configs/esft_math_experts.json")
        >>> # Result: {0: [2, 5, 12], 1: [8, 15, 27], ...}
    """
    selection_path = Path(selection_path)

    if not selection_path.exists():
        raise FileNotFoundError(
            f"Expert assignment not found: {selection_path}\n"
            f"Subnet owner must run ESFT and publish the selection file."
        )

    with open(selection_path) as f:
        data = json.load(f)

    # Extract selected experts from layer_stats
    selected_experts = {}
    for layer_key, stats in data.get("layer_stats", {}).items():
        layer_idx = int(layer_key.replace("layer_", ""))
        selected_experts[layer_idx] = stats["selected_experts"]

    logger.info(
        f"Loaded expert assignment from {selection_path}\n"
        f"  Method: {data.get('method')}\n"
        f"  Threshold: {data.get('threshold')}\n"
        f"  Avg experts/layer: {data.get('avg_experts_per_layer'):.1f}"
    )

    return selected_experts


def apply_expert_assignment(
    model: nn.Module,
    selected_experts: Dict[int, List[int]],
    num_experts: int = 128,
    shared_expert_id: int = 127,
):
    """
    Apply expert assignment to model (freeze non-selected experts).

    This is what miners call after loading subnet owner's published assignment.

    Args:
        model: MoE model
        selected_experts: {layer_idx: [expert_ids]} from load_expert_assignment()
        num_experts: Total number of experts (default: 128)
        shared_expert_id: Shared expert ID (default: 127)

    Example (Miner):
        >>> # Load published assignment
        >>> selected = load_expert_assignment("configs/esft_math_experts.json")
        >>>
        >>> # Apply to model
        >>> apply_expert_assignment(model, selected)
        >>>
        >>> # Train (only selected experts are trainable)
        >>> train(model, dataloader)
    """
    # Create selector for freezing logic
    config = ESFTConfig(
        method="token",  # Doesn't matter, we're just using freeze logic
        threshold=0.2,
    )
    selector = ESFTExpertSelector(model, config, num_experts, shared_expert_id)

    # Apply freezing
    selector.freeze_non_selected_experts(selected_experts)

    logger.info("Expert assignment applied successfully")


def get_trainable_expert_count(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable vs total expert parameters.

    Args:
        model: MoE model

    Returns:
        (trainable_params, total_params)

    Example:
        >>> trainable, total = get_trainable_expert_count(model)
        >>> print(f"Training {trainable/total*100:.1f}% of expert parameters")
    """
    trainable_params = 0
    total_params = 0

    # Get MoE layers
    if hasattr(model, 'language_model'):
        layers = model.language_model.layers
    elif hasattr(model, 'model'):
        layers = model.model.layers
    else:
        raise ValueError("Cannot find model layers")

    for layer in layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            for expert in layer.mlp.experts.values():
                for param in expert.parameters():
                    total_params += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()

    return trainable_params, total_params
