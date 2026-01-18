"""
Shared Expert MoE Layer for Qwen3-VL

Adds a shared expert FFN that processes every token in parallel with routed experts.
This ensures baseline capabilities while allowing expert specialization.

Architecture:
    Input → [Routed Experts (top-k)] + [Shared Expert (always-on)] → Combined Output

Benefits:
- Maintains general knowledge (shared expert handles common patterns)
- Allows specialization (routed experts focus on domain-specific patterns)
- Prevents catastrophic forgetting during fine-tuning

Implementation Modes:
1. "designated" - Use one existing expert as shared (no new params)
   - Expert is forced to always be selected via logit assignment
   - Works even when masked by pool routing (-inf + bias doesn't work, so we assign)
   - Memory efficient: no new parameters

2. "new_ffn" - Add separate shared expert FFN layer (more params)
   - New FFN processes all tokens in parallel with routed experts
   - Outputs are combined: routed_output + shared_output
   - More explicit separation of shared vs specialized knowledge

Critical Design Decisions:

A) Designated Expert Forcing (lines 173-189):
   - Must use ASSIGNMENT not addition: -inf + bias = -inf in floating point
   - max_logit computed AFTER masking (critical ordering)
   - Defensive: Falls back to 0.0 if all experts masked (buggy config)
   - Formula: router_logits[:, designated_id] = max(router_logits) + bias

B) Operation Ordering (lines 164-189):
   1. Compute router logits
   2. Add jitter noise (before masking)
   3. Apply expert pool mask (sets excluded experts to -1e9)
   4. Force designated expert (overrides mask via assignment)
   5. Top-k selection
   This ordering ensures designated expert always selected even if masked.

C) Validation (lines 198-208):
   - Optional (default: False) to avoid slowdown
   - Sampled every 100 forward passes
   - Only during training (when gradients enabled)
   - Warns if designated expert not selected (should never happen if implemented correctly)

D) Numerical Stability:
   - Uses finite bias values (not +inf) to avoid NaN in softmax
   - max_logit + bias keeps values finite
   - Handles -inf masks gracefully via assignment

Known Limitations (must be handled in training code):
1. Load-balance loss: Must exclude designated expert from aux loss computation
   - Otherwise router gets penalized for "overusing" the forced expert
   - Pattern (epsilon-protected):
     probs = router_probs.clone()
     probs[..., designated_id] = 0.0
     den = probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # Prevent div by zero
     probs = probs / den
     aux_loss = compute_load_balance_loss(probs)
   - Alternative: If using top-k assignments, compute balance stats from
     selected_experts excluding designated_id (simpler, avoids probability renorm)

2. Capacity limiting: If Qwen3-VL uses expert capacity, designated expert may be
   - Always selected but not always process all tokens (overflow drops tokens)
   - Verify via dropped_tokens counters (if exposed) or dispatch masks
   - Invariant: tokens_dispatched <= tokens_routed, dropped = difference
   - Check for overflow during validation to ensure "always-on" isn't undermined
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextSparseMoeBlock,
    Qwen3VLMoeTextMLP,
)


class SharedExpertMLP(nn.Module):
    """
    Shared expert FFN that processes all tokens.

    This is a standard FFN layer similar to individual experts but runs on every token.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        # Standard FFN: up projection → activation → down projection
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()  # SwiGLU activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared expert.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # SwiGLU: (gate_proj(x) * silu) ⊙ up_proj(x)
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate = gate_output * up_output
        output = self.down_proj(intermediate)
        return output


class Qwen3VLSharedExpertMoeBlock(nn.Module):
    """
    MoE block with both routed experts and a shared expert.

    Supports two modes:
    1. "designated" - Use one existing expert as shared (no new params)
    2. "new_ffn" - Add new FFN layer as shared expert (more params)

    Combines:
    1. Top-k routed experts (sparse activation)
    2. Shared expert (dense activation on all tokens)

    Final output = routed_output + shared_expert_output
    """

    def __init__(
        self,
        config,
        original_moe_block: Optional[Qwen3VLMoeTextSparseMoeBlock] = None,
        shared_expert_mode: str = "designated",
        designated_expert_id: int = None,
        bias_strength: float = 10.0,
        validate_routing: bool = False,  # Enable runtime validation (for debugging)
    ):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.shared_expert_mode = shared_expert_mode
        self.designated_expert_id = designated_expert_id
        self.bias_strength = bias_strength
        self.validate_routing = validate_routing
        self._forward_count = 0  # For sampled validation

        if original_moe_block is not None:
            # Copy existing MoE components
            self.gate = original_moe_block.gate
            self.experts = original_moe_block.experts
            # Check if experts is Qwen3VLMoeTextExperts (uses tensor params) or ModuleList
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts
            self.use_expert_tensors = isinstance(self.experts, Qwen3VLMoeTextExperts)
        else:
            # Initialize from scratch
            self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            self.experts = nn.ModuleList([
                Qwen3VLMoeTextMLP(config) for _ in range(config.num_experts)
            ])
            self.use_expert_tensors = False

        # Add shared expert based on mode
        if shared_expert_mode == "new_ffn":
            # Mode 1: Add new FFN layer (more params, but explicit shared expert)
            self.shared_expert = SharedExpertMLP(config)
            # Optional: Learnable weight for shared expert contribution
            self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
        elif shared_expert_mode == "designated":
            # Mode 2: Use existing expert as shared (no new params)
            # Designated expert will be forced via logit bias
            if designated_expert_id is None:
                # Default to last expert
                self.designated_expert_id = self.num_experts - 1
            else:
                assert 0 <= designated_expert_id < self.num_experts, \
                    f"designated_expert_id {designated_expert_id} out of range [0, {self.num_experts})"
                self.designated_expert_id = designated_expert_id

            self.shared_expert = None  # No separate shared expert
            self.shared_expert_gate = None
            print(f"  Using expert {self.designated_expert_id} as designated shared expert (bias={bias_strength})")
        else:
            raise ValueError(f"Unknown shared_expert_mode: {shared_expert_mode}. Use 'designated' or 'new_ffn'")

        # Load balancing
        self.jitter_noise = config.router_jitter_noise if hasattr(config, 'router_jitter_noise') else 0.0

        # Initialize current_expert_mask (can be set by MaskedRoutingWrapper)
        self.current_expert_mask = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,  # For masked routing
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with routed + shared experts.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            expert_mask: Optional mask to restrict routing to specific experts
                        [num_experts] with 1=allowed, 0=forbidden

        Returns:
            (output, router_logits):
                output: Combined output from routed + shared experts
                router_logits: Raw router scores for load balancing loss
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Flatten for routing: [batch_size * seq_len, hidden_dim]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # ===================================================================
        # 1. ROUTED EXPERTS (Sparse, Top-K)
        # ===================================================================

        # Router computes expert scores
        router_logits = self.gate(hidden_states_flat)  # [B*T, num_experts]

        # Add jitter noise during training (for load balancing)
        # NOTE: Apply noise BEFORE masking/designated expert forcing
        if self.training and self.jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise

        # Apply expert mask if provided (for pool-specific training)
        # Fallback to current_expert_mask set by MaskedRoutingWrapper
        if expert_mask is None:
            expert_mask = self.current_expert_mask

        if expert_mask is not None:
            # CRITICAL FIX: Mask is [0 for allowed, -1e9 for forbidden]
            # Apply ADDITIVELY (not multiplicatively - that would be fatal!)
            # Wrong: router_logits + expert_mask * -1e9  # (-inf * -1e9 = +inf, NaN in softmax!)
            # Correct:
            router_logits = router_logits + expert_mask.unsqueeze(0)

        # For "designated" mode: Force designated expert to be selectable
        # CRITICAL: Must use assignment, not addition, since -inf + bias = -inf
        if self.shared_expert_mode == "designated":
            # Find max logit across experts AFTER masking (critical ordering)
            max_logit = router_logits.max(dim=-1, keepdim=True).values

            # Defensive: Handle edge case where all experts are masked (buggy config)
            # If max_logit is -inf, fall back to finite value with correct dtype/device
            max_logit_finite = torch.where(
                torch.isinf(max_logit),
                torch.zeros_like(max_logit, dtype=router_logits.dtype, device=router_logits.device),
                max_logit
            )

            # Force designated expert logit to be max + margin (guarantees top-k selection)
            # Use assignment to override even -inf masks
            router_logits[:, self.designated_expert_id] = max_logit_finite.squeeze(-1) + self.bias_strength

        # Select top-k experts
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Validation: In designated mode, verify designated expert is always selected
        # Only validate if explicitly enabled, and sample every 100 forward passes to avoid slowdown
        if self.shared_expert_mode == "designated" and self.validate_routing and torch.is_grad_enabled():
            self._forward_count += 1
            if self._forward_count % 100 == 0:  # Sample every 100 forward passes
                designated_selected = (selected_experts == self.designated_expert_id).any(dim=-1)
                if not designated_selected.all():
                    num_failed = (~designated_selected).sum().item()
                    # Only print on rank 0 in distributed training to avoid spam
                    import torch.distributed as dist
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(f"WARNING [step {self._forward_count}]: Designated expert {self.designated_expert_id} "
                              f"not selected for {num_failed}/{designated_selected.numel()} tokens. "
                              f"Consider increasing bias_strength (current: {self.bias_strength})")

        # Normalize routing weights (optional, depends on your preference)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Process tokens through routed experts
        if self.use_expert_tensors:
            # Use Qwen3VLMoeTextExperts forward method (tensor-based experts)
            # Need to reconstruct full routing weights [B*T, num_experts] from top-k
            full_routing_weights = torch.zeros(
                (hidden_states_flat.shape[0], self.num_experts),
                dtype=routing_weights.dtype,
                device=routing_weights.device
            )
            # Scatter top-k weights into full tensor
            full_routing_weights.scatter_(1, selected_experts, routing_weights)
            routed_output = self.experts(hidden_states_flat, full_routing_weights, selected_experts)
        else:
            # Use ModuleList (module-based experts) - manual iteration
            routed_output = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

            # One-hot encode selected experts for scatter
            expert_mask_onehot = F.one_hot(selected_experts, num_classes=self.num_experts)
            expert_mask_onehot = expert_mask_onehot.permute(2, 1, 0)  # [num_experts, top_k, B*T]

            # Process tokens through selected experts
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]

                # Get tokens routed to this expert
                idx, top_x = torch.where(expert_mask_onehot[expert_idx])

                if top_x.shape[0] == 0:
                    continue  # No tokens routed to this expert

                # Process selected tokens
                current_hidden = hidden_states_flat[top_x]
                current_output = expert_layer(current_hidden)

                # Weight by routing weights
                weighted_output = current_output * routing_weights[top_x, idx, None]

                # Accumulate into final output
                routed_output.index_add_(0, top_x, weighted_output)

        # ===================================================================
        # 2. SHARED EXPERT (Dense, All Tokens) - Only for "new_ffn" mode
        # ===================================================================

        if self.shared_expert_mode == "new_ffn":
            # Mode 1: Add output from separate shared expert FFN
            shared_output = self.shared_expert(hidden_states_flat)

            # Optional: Learnable gating for shared expert contribution
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states_flat))
            shared_output = shared_output * shared_gate
        else:
            # Mode 2: "designated" mode - shared expert is already in routed output
            # (designated expert was biased to always be selected in top-k)
            shared_output = 0.0

        # ===================================================================
        # 3. COMBINE OUTPUTS
        # ===================================================================

        # Final output = routed experts + shared expert (if new_ffn mode)
        final_output = routed_output + shared_output

        # Reshape back to [batch_size, seq_len, hidden_dim]
        final_output = final_output.view(batch_size, sequence_length, hidden_dim)

        return final_output, router_logits


def convert_qwen3vl_to_shared_expert_moe(
    model,
    freeze_original_experts: bool = False,
    shared_expert_mode: str = "designated",
    designated_expert_id: int = None,
    bias_strength: float = 10.0,
):
    """
    Convert a Qwen3-VL model to use shared expert MoE layers.

    Args:
        model: Qwen3VLForConditionalGeneration or similar
        freeze_original_experts: If True, freeze routed experts (only train shared expert)
        shared_expert_mode: "designated" (use existing expert) or "new_ffn" (add new layer)
        designated_expert_id: Which expert to use as shared (default: last expert)
        bias_strength: Logit bias for designated expert (default: 10.0)

    Returns:
        Modified model with shared expert MoE layers
    """
    print("Converting Qwen3-VL to Shared Expert MoE architecture...")
    print(f"  Mode: {shared_expert_mode}")

    # Get text config (where MoE params are stored)
    text_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config

    if shared_expert_mode == "designated":
        expert_id = designated_expert_id if designated_expert_id is not None else text_config.num_experts - 1
        print(f"  Designated expert: {expert_id} (bias strength: {bias_strength})")

    # Access language model layers
    # For Qwen3-VL: model.model.language_model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        language_model = model.model.language_model
    elif hasattr(model, 'model'):
        language_model = model.model
    else:
        language_model = model

    if not hasattr(language_model, 'layers'):
        raise ValueError(f"Cannot find model layers. Model structure: {type(language_model)}")

    moe_layer_count = 0

    # Iterate through transformer layers
    for layer_idx, layer in enumerate(language_model.layers):
        # Check if this layer has MoE
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, Qwen3VLMoeTextSparseMoeBlock):
            original_moe = layer.mlp

            # Replace with shared expert version
            new_moe = Qwen3VLSharedExpertMoeBlock(
                config=text_config,  # Use text_config, not model.config
                original_moe_block=original_moe,
                shared_expert_mode=shared_expert_mode,
                designated_expert_id=designated_expert_id,
                bias_strength=bias_strength,
            )

            # Optionally freeze original experts
            if freeze_original_experts:
                for param in new_moe.gate.parameters():
                    param.requires_grad = False
                for expert in new_moe.experts:
                    for param in expert.parameters():
                        param.requires_grad = False

                print(f"  Layer {layer_idx}: Froze routed experts, only training shared expert")

            layer.mlp = new_moe
            moe_layer_count += 1

    print(f"\nConversion complete! Modified {moe_layer_count} MoE layers.")
    if shared_expert_mode == "designated":
        print(f"  - {text_config.num_experts} routed experts (top-{text_config.num_experts_per_tok} activated)")
        print(f"  - Expert {designated_expert_id if designated_expert_id is not None else text_config.num_experts - 1} designated as shared (biased to always be selected)")
    else:
        print(f"  - {text_config.num_experts} routed experts (top-{text_config.num_experts_per_tok} activated)")
        print(f"  - 1 new shared expert FFN (always activated)")

    return model


def load_qwen3vl_with_shared_experts(
    model_path: str = "Qwen/Qwen3-VL-30B-A3B-Thinking",
    freeze_routed_experts: bool = False,
    device_map: str = "auto",
):
    """
    Load Qwen3-VL model and add shared experts.

    Args:
        model_path: HuggingFace model path
        freeze_routed_experts: Whether to freeze original experts
        device_map: Device mapping for multi-GPU

    Returns:
        Model with shared expert MoE architecture
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"Loading {model_path}...")

    # Load base model
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Convert to shared expert architecture
    model = convert_qwen3vl_to_shared_expert_moe(
        model,
        freeze_original_experts=freeze_routed_experts
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    return model, processor


# ===================================================================
# MASKED ROUTING FOR EXPERT POOLS
# ===================================================================

class MaskedRoutingWrapper:
    """
    Wrapper to enable task-specific expert pool routing.

    Usage:
        wrapper = MaskedRoutingWrapper(model, expert_pools)
        with wrapper.route_to_pool("ROBOTICS"):
            outputs = model(**batch)
    """

    def __init__(self, model, expert_pool_config: dict):
        """
        Args:
            model: Qwen3-VL model with shared expert MoE
            expert_pool_config: Dict like {"ROBOTICS": [0, 199], "MATH": [200, 299], ...}
        """
        self.model = model
        self.expert_pool_config = expert_pool_config
        self.current_mask = None

    def route_to_pool(self, task_name: str):
        """Context manager to route to specific expert pool."""
        return RoutingContext(self, task_name)

    def _create_expert_mask(self, allowed_experts: list) -> torch.Tensor:
        """
        Create mask tensor for routing.

        Args:
            allowed_experts: List of allowed expert indices

        Returns:
            Mask tensor [num_experts] with 0=allowed, -1e9=forbidden
        """
        # Get num_experts from text_config
        text_config = self.model.config.text_config if hasattr(self.model.config, 'text_config') else self.model.config
        num_experts = text_config.num_experts

        # Use -1e9 instead of -inf for better numerical stability
        # (avoids potential NaN in softmax, though should be fine either way)
        mask = torch.full((num_experts,), -1e9)
        mask[allowed_experts] = 0.0
        return mask

    def apply_mask(self, task_name: str):
        """Apply expert mask for given task."""
        if task_name not in self.expert_pool_config:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.expert_pool_config.keys())}")

        allowed_experts = self.expert_pool_config[task_name]
        self.current_mask = self._create_expert_mask(allowed_experts)

        # Get language model layers (same logic as in conversion function)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            language_model = self.model.model.language_model
        elif hasattr(self.model, 'model'):
            language_model = self.model.model
        else:
            language_model = self.model

        # Apply mask to all MoE layers
        for layer in language_model.layers:
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, Qwen3VLSharedExpertMoeBlock):
                # Get device from gate (which always exists)
                device = layer.mlp.gate.weight.device
                layer.mlp.current_expert_mask = self.current_mask.to(device)

    def remove_mask(self):
        """Remove expert routing mask."""
        self.current_mask = None

        # Get language model layers (same logic as in conversion function)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            language_model = self.model.model.language_model
        elif hasattr(self.model, 'model'):
            language_model = self.model.model
        else:
            language_model = self.model

        for layer in language_model.layers:
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, Qwen3VLSharedExpertMoeBlock):
                layer.mlp.current_expert_mask = None


class RoutingContext:
    """Context manager for masked routing."""

    def __init__(self, wrapper: MaskedRoutingWrapper, task_name: str):
        self.wrapper = wrapper
        self.task_name = task_name

    def __enter__(self):
        self.wrapper.apply_mask(self.task_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wrapper.remove_mask()


# ===================================================================
# EXAMPLE USAGE
# ===================================================================

if __name__ == "__main__":
    # Example: Load model with shared experts
    model, processor = load_qwen3vl_with_shared_experts(
        model_path="Qwen/Qwen3-VL-30B-A3B-Thinking",
        freeze_routed_experts=False,  # Train all experts
        device_map="auto",
    )

    # Example: Masked routing for robotics training
    expert_pools = {
        "ROBOTICS": list(range(0, 200)),      # Experts 0-199 for robotics
        "MATH": list(range(200, 300)),        # Experts 200-299 for math
        "CODE": list(range(300, 400)),        # Experts 300-399 for code
        "GENERAL": list(range(400, 512)),     # Experts 400-511 for general
    }

    routing_wrapper = MaskedRoutingWrapper(model, expert_pools)

    # Training loop example
    robotics_batch = {...}  # Your robotics batch with images + text

    with routing_wrapper.route_to_pool("ROBOTICS"):
        outputs = model(**robotics_batch)
        loss = outputs.loss
        loss.backward()

    print("Robotics expert pool training complete!")