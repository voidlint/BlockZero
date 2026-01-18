"""Expert-Specialized Fine-Tuning (ESFT) for MoE models.

Qwen3-MoE has 128 experts with 8 active per token, NO shared experts.
This means only 6.25% of experts get gradients per forward pass - causing tiny gradients.

ESFT solves this by concentrating training on a subset of experts:

Method 1: Partition-based (recommended for distributed training)
    - Partition 128 experts into 4 task sets (32 experts each)
    - Each miner trains only their partition: 8/32 = 25% gradient coverage
    - 4x stronger gradients than training all 128!

Method 2: Route-based (recommended for single miner)
    - Measure which experts route to your training data
    - Train only top-K experts that receive your data
    - Concentrates gradients on relevant experts

Usage:
    from mycelia.shared.esft import apply_esft_partition, measure_expert_routing

    # Method 1: Partition-based
    trainable = apply_esft_partition(model, partition_id=0, experts_per_partition=32)

    # Method 2: Route-based
    routing = measure_expert_routing(model, dataloader, num_batches=100)
    trainable = apply_esft_routing(model, routing, top_k=32)
"""

from collections import Counter

import torch

from mycelia.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


def get_moe_layers(model):
    """
    Find all MoE layers in the model.

    Returns list of (layer_idx, layer_module) tuples for layers that have MoE.
    """
    moe_layers = []

    # Try different model architectures
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        logger.warning("Could not find layers in model architecture")
        return []

    for layer_idx, layer in enumerate(layers):
        # Check for MoE components
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        # Check for gate/router
        has_gate = hasattr(mlp, "gate") or hasattr(mlp, "router")

        # Check for experts
        has_experts = hasattr(mlp, "experts") or hasattr(mlp, "expert_weights")

        if has_gate or has_experts:
            moe_layers.append((layer_idx, layer))

    return moe_layers


def measure_expert_routing(model, dataloader, num_batches: int = 100, device=None):
    """
    Measure which experts are actually selected for your training data.

    Args:
        model: The MoE model
        dataloader: Training dataloader
        num_batches: Number of batches to measure (more = more accurate)
        device: Device to run on

    Returns:
        Counter mapping "layer_{idx}_expert_{id}" to selection counts
    """
    if device is None:
        device = next(model.parameters()).device

    expert_counts = Counter()
    moe_layers = get_moe_layers(model)

    if not moe_layers:
        logger.warning("No MoE layers found - cannot measure routing")
        return expert_counts

    logger.info(f"Measuring expert routing on {num_batches} batches across {len(moe_layers)} MoE layers")

    # Store hooks to capture routing decisions
    routing_data = {}
    hooks = []

    def make_routing_hook(layer_idx):
        """Create a hook to capture routing decisions for a layer."""

        def hook(module, input, output):
            # The output format varies by implementation
            # Try to extract expert indices from output
            if isinstance(output, tuple) and len(output) >= 2:
                # Common format: (hidden_states, router_logits, expert_indices)
                if len(output) >= 3 and output[2] is not None:
                    expert_indices = output[2]
                    if layer_idx not in routing_data:
                        routing_data[layer_idx] = []
                    routing_data[layer_idx].append(expert_indices.detach().cpu())

        return hook

    # Register hooks on MoE layers
    for layer_idx, layer in moe_layers:
        mlp = layer.mlp
        if hasattr(mlp, "gate"):
            hook = mlp.gate.register_forward_hook(make_routing_hook(layer_idx))
            hooks.append(hook)
        elif hasattr(mlp, "router"):
            hook = mlp.router.register_forward_hook(make_routing_hook(layer_idx))
            hooks.append(hook)

    # Run forward passes to collect routing data
    model.eval()
    batches_processed = 0

    try:
        with torch.no_grad():
            for batch in dataloader:
                if batches_processed >= num_batches:
                    break

                # Move batch to device
                batch_device = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_device[key] = value.to(device)
                    else:
                        batch_device[key] = value

                try:
                    # Forward pass to trigger hooks
                    _ = model(**batch_device)
                    batches_processed += 1
                except Exception as e:
                    logger.warning(f"Error in forward pass during routing measurement: {e}")
                    continue

                # Process captured routing data
                for layer_idx, indices_list in routing_data.items():
                    for indices in indices_list:
                        # Flatten and count unique expert selections
                        flat_indices = indices.view(-1)
                        for expert_id in flat_indices.unique().tolist():
                            key = f"layer_{layer_idx}_expert_{expert_id}"
                            expert_counts[key] += int((flat_indices == expert_id).sum())

                routing_data.clear()

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    # If hooks didn't capture data, try alternative method
    if not expert_counts:
        logger.info("Hook-based routing measurement failed, trying alternative method")
        expert_counts = _measure_routing_alternative(model, dataloader, num_batches, device)

    # Log results
    if expert_counts:
        _log_routing_statistics(expert_counts, moe_layers)
    else:
        logger.warning("Could not measure expert routing - will use default freezing")

    return expert_counts


def _measure_routing_alternative(model, dataloader, num_batches, device):
    """
    Alternative routing measurement that captures gate outputs directly.
    """
    expert_counts = Counter()
    moe_layers = get_moe_layers(model)

    model.eval()
    batches_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if batches_processed >= num_batches:
                break

            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            try:
                # Get hidden states through embedding layer
                if hasattr(model, "language_model"):
                    embed = model.language_model.embed_tokens
                elif hasattr(model, "model"):
                    embed = model.model.embed_tokens
                else:
                    continue

                input_ids = batch_device.get("input_ids")
                if input_ids is None:
                    continue

                hidden_states = embed(input_ids)

                # Process through each layer to get routing decisions
                for layer_idx, layer in moe_layers:
                    mlp = layer.mlp
                    gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)

                    if gate is None:
                        continue

                    # Get gate logits
                    # Flatten hidden states for gate input
                    flat_hidden = hidden_states.view(-1, hidden_states.size(-1))

                    try:
                        if hasattr(gate, "weight"):
                            gate_logits = torch.matmul(flat_hidden, gate.weight.T)
                        else:
                            gate_logits = gate(flat_hidden)

                        # Get top-k expert selections (typically k=2 for MoE)
                        k = min(2, gate_logits.size(-1))
                        top_experts = torch.topk(gate_logits, k, dim=-1).indices

                        for expert_id in top_experts.view(-1).unique().tolist():
                            key = f"layer_{layer_idx}_expert_{expert_id}"
                            expert_counts[key] += int((top_experts == expert_id).sum())
                    except Exception:
                        continue

                batches_processed += 1

            except Exception as e:
                logger.debug(f"Alternative routing measurement error: {e}")
                continue

    return expert_counts


def _log_routing_statistics(expert_counts, moe_layers):
    """Log expert routing statistics."""
    # Group by layer
    per_layer = {}
    for key, count in expert_counts.items():
        parts = key.split("_")
        layer_idx = int(parts[1])
        expert_id = int(parts[3])

        if layer_idx not in per_layer:
            per_layer[layer_idx] = []
        per_layer[layer_idx].append((expert_id, count))

    # Log top experts per layer
    for layer_idx in sorted(per_layer.keys())[:5]:  # First 5 layers
        experts = per_layer[layer_idx]
        top_5 = sorted(experts, key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Layer {layer_idx} top experts: {top_5}")

    total_selections = sum(expert_counts.values())
    logger.info(f"Total expert selections measured: {total_selections}")


def apply_esft(model, expert_routing: Counter, top_k: int = 6):
    """
    Apply ESFT: Freeze all experts except top-K most-used per layer.

    Args:
        model: The MoE model
        expert_routing: Counter from measure_expert_routing()
        top_k: Number of experts to keep trainable per layer

    Returns:
        Dict mapping layer_idx -> set of trainable expert IDs
    """
    if not expert_routing:
        logger.warning("No routing data - ESFT cannot be applied, using default freezing")
        return {}

    # Parse routing data into per-layer expert rankings
    per_layer_experts = {}
    for key, count in expert_routing.items():
        if "layer_" in key and "expert_" in key:
            parts = key.split("_")
            layer_idx = int(parts[1])
            expert_id = int(parts[3])

            if layer_idx not in per_layer_experts:
                per_layer_experts[layer_idx] = []
            per_layer_experts[layer_idx].append((expert_id, count))

    # Select top-K experts per layer
    trainable_experts = {}
    for layer_idx, experts in per_layer_experts.items():
        top_experts = sorted(experts, key=lambda x: x[1], reverse=True)[:top_k]
        trainable_experts[layer_idx] = set(e[0] for e in top_experts)

    logger.info(f"ESFT selected experts per layer: {trainable_experts}")

    # Apply freezing
    moe_layers = get_moe_layers(model)
    total_params = 0
    trainable_params = 0
    frozen_experts = 0
    trained_experts = 0

    for layer_idx, layer in moe_layers:
        mlp = layer.mlp
        layer_trainable = trainable_experts.get(layer_idx, set())

        # Handle different expert storage formats
        experts = None
        if hasattr(mlp, "experts"):
            experts = mlp.experts
        elif hasattr(mlp, "expert_weights"):
            experts = mlp.expert_weights

        if experts is not None:
            # experts could be a dict, list, or ModuleList
            if isinstance(experts, dict):
                expert_items = experts.items()
            elif hasattr(experts, "__iter__"):
                expert_items = enumerate(experts)
            else:
                continue

            for expert_id, expert in expert_items:
                expert_id = int(expert_id) if not isinstance(expert_id, int) else expert_id
                is_trainable = expert_id in layer_trainable

                for param in expert.parameters():
                    total_params += param.numel()
                    if is_trainable:
                        param.requires_grad = True
                        trainable_params += param.numel()
                        trained_experts += 1
                    else:
                        param.requires_grad = False
                        frozen_experts += 1

        # Always freeze the router/gate (as per ESFT paper)
        gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
        if gate is not None:
            for param in gate.parameters():
                param.requires_grad = False
                total_params += param.numel()

    # Freeze non-MoE layers (embeddings, layer norms, etc.)
    for name, param in model.named_parameters():
        if "expert" not in name.lower() and "mlp" not in name.lower():
            param.requires_grad = False

    # Report statistics
    trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0
    logger.info(
        "ESFT applied",
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_pct=f"{trainable_pct:.1f}%",
        trained_experts=trained_experts,
        frozen_experts=frozen_experts,
    )

    return trainable_experts


def get_esft_summary(model):
    """Get a summary of which parameters are trainable after ESFT."""
    trainable_by_type = Counter()
    frozen_by_type = Counter()

    for name, param in model.named_parameters():
        # Categorize parameter
        if "expert" in name.lower():
            ptype = "expert"
        elif "gate" in name.lower() or "router" in name.lower():
            ptype = "router"
        elif "embed" in name.lower():
            ptype = "embedding"
        elif "norm" in name.lower():
            ptype = "norm"
        elif "attn" in name.lower() or "attention" in name.lower():
            ptype = "attention"
        else:
            ptype = "other"

        if param.requires_grad:
            trainable_by_type[ptype] += param.numel()
        else:
            frozen_by_type[ptype] += param.numel()

    return {
        "trainable": dict(trainable_by_type),
        "frozen": dict(frozen_by_type),
    }


def measure_expert_overlap(model, dataloader, num_batches: int = 50, device=None):
    """
    Measure which experts are used together (co-occurrence).

    If experts are frequently used together, they have gradient coupling which
    can cause interference. This helps verify that overlap is properly handled.

    Args:
        model: The MoE model
        dataloader: Training dataloader
        num_batches: Number of batches to measure
        device: Device to run on

    Returns:
        Dict with co-occurrence statistics per layer
    """
    if device is None:
        device = next(model.parameters()).device

    expert_co_occurrence = Counter()
    moe_layers = get_moe_layers(model)

    if not moe_layers:
        logger.warning("No MoE layers found - cannot measure overlap")
        return {}

    model.eval()
    batches_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            if batches_processed >= num_batches:
                break

            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            try:
                # Get hidden states through embedding layer
                if hasattr(model, "language_model"):
                    embed = model.language_model.embed_tokens
                elif hasattr(model, "model"):
                    embed = model.model.embed_tokens
                else:
                    continue

                input_ids = batch_device.get("input_ids")
                if input_ids is None:
                    continue

                hidden_states = embed(input_ids)

                # Process through each layer to get routing decisions
                for layer_idx, layer in moe_layers:
                    mlp = layer.mlp
                    gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)

                    if gate is None:
                        continue

                    # Get gate logits
                    flat_hidden = hidden_states.view(-1, hidden_states.size(-1))

                    try:
                        if hasattr(gate, "weight"):
                            gate_logits = torch.matmul(flat_hidden, gate.weight.T)
                        else:
                            gate_logits = gate(flat_hidden)

                        # Get top-2 selected experts (typical for MoE)
                        k = min(2, gate_logits.size(-1))
                        top_experts = torch.topk(gate_logits, k, dim=-1).indices

                        # Track co-occurrence (which expert pairs are used together)
                        for token_idx in range(top_experts.size(0)):
                            if k >= 2:
                                expert_pair = tuple(sorted(top_experts[token_idx].tolist()))
                                key = f"layer_{layer_idx}_pair_{expert_pair}"
                                expert_co_occurrence[key] += 1
                    except Exception:
                        continue

                batches_processed += 1

            except Exception as e:
                logger.debug(f"Expert overlap measurement error: {e}")
                continue

    # Log most common expert pairs per layer
    if expert_co_occurrence:
        _log_overlap_statistics(expert_co_occurrence, moe_layers)

    return dict(expert_co_occurrence)


def _log_overlap_statistics(expert_co_occurrence, moe_layers):
    """Log expert co-occurrence statistics."""
    logger.info("=== EXPERT OVERLAP ANALYSIS ===")

    for layer_idx, _ in moe_layers[:5]:  # First 5 layers
        layer_pairs = {k: v for k, v in expert_co_occurrence.items() if f"layer_{layer_idx}_" in k}
        if layer_pairs:
            top_pairs = sorted(layer_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
            pairs_formatted = [(k.split("pair_")[1], v) for k, v in top_pairs]
            logger.info(f"Layer {layer_idx} top expert pairs: {pairs_formatted}")

    total_co_occurrences = sum(expert_co_occurrence.values())
    logger.info(f"Total expert pair co-occurrences: {total_co_occurrences}")


def identify_shared_experts(expert_routing: Counter, num_shared: int = 2):
    """
    Identify which experts should be designated as 'shared' experts.

    Shared experts are those used most consistently across all tokens/batches.
    They typically handle common knowledge (grammar, basic reasoning).

    Args:
        expert_routing: Counter from measure_expert_routing()
        num_shared: Number of shared experts to identify

    Returns:
        Set of expert IDs that should be shared (per-layer dict)
    """
    if not expert_routing:
        return {}

    # Aggregate expert usage across all layers
    per_layer_usage = {}
    for key, count in expert_routing.items():
        if "layer_" in key and "expert_" in key:
            parts = key.split("_")
            layer_idx = int(parts[1])
            expert_id = int(parts[3])

            if layer_idx not in per_layer_usage:
                per_layer_usage[layer_idx] = Counter()
            per_layer_usage[layer_idx][expert_id] = count

    # For each layer, identify the most-used experts as "shared"
    shared_experts = {}
    for layer_idx, usage in per_layer_usage.items():
        # Top num_shared experts by usage are designated as shared
        top_experts = usage.most_common(num_shared)
        shared_experts[layer_idx] = set(expert_id for expert_id, _ in top_experts)

    logger.info(f"Identified shared experts (top {num_shared} per layer)")
    for layer_idx in sorted(shared_experts.keys())[:5]:
        logger.info(f"  Layer {layer_idx}: {shared_experts[layer_idx]}")

    return shared_experts


def apply_esft_with_overlap(
    model,
    expert_routing: Counter,
    top_k: int = 6,
    num_shared: int = 2,
    shared_expert_ids: list[int] | None = None,
):
    """
    Apply overlap-aware ESFT: Train shared experts + top-K task-specific experts.

    This combines the DeepSeekMoE shared/routed expert separation with ESFT's
    data-driven expert selection for optimal gradient flow.

    Architecture:
        Shared Experts (always trainable):
            - Handle common knowledge (grammar, reasoning)
            - Either auto-identified or manually specified

        Task Experts (selected by routing):
            - Top-K experts that route to your training data
            - Excludes shared experts to avoid double-counting

        Frozen:
            - All other experts
            - Router/gate (frozen per ESFT paper)
            - Non-MoE layers (embeddings, attention, norms)

    Args:
        model: The MoE model
        expert_routing: Counter from measure_expert_routing()
        top_k: Number of task-specific experts to train per layer
        num_shared: Number of shared experts per layer (if shared_expert_ids is None)
        shared_expert_ids: Explicit list of shared expert IDs (same for all layers)

    Returns:
        Dict with 'shared' and 'task' expert sets per layer
    """
    if not expert_routing:
        logger.warning("No routing data - overlap-aware ESFT cannot be applied")
        return {}

    # Step 1: Identify shared experts
    if shared_expert_ids is not None:
        # Use explicitly specified shared expert IDs
        moe_layers = get_moe_layers(model)
        shared_experts = {layer_idx: set(shared_expert_ids) for layer_idx, _ in moe_layers}
        logger.info(f"Using explicit shared expert IDs: {shared_expert_ids}")
    else:
        # Auto-identify shared experts from routing data
        shared_experts = identify_shared_experts(expert_routing, num_shared)

    # Step 2: Parse routing data into per-layer expert rankings
    per_layer_experts = {}
    for key, count in expert_routing.items():
        if "layer_" in key and "expert_" in key:
            parts = key.split("_")
            layer_idx = int(parts[1])
            expert_id = int(parts[3])

            if layer_idx not in per_layer_experts:
                per_layer_experts[layer_idx] = []
            per_layer_experts[layer_idx].append((expert_id, count))

    # Step 3: Select top-K task experts (excluding shared ones)
    task_experts = {}
    for layer_idx, experts in per_layer_experts.items():
        layer_shared = shared_experts.get(layer_idx, set())

        # Filter out shared experts and sort by routing count
        non_shared = [(eid, cnt) for eid, cnt in experts if eid not in layer_shared]
        top_task = sorted(non_shared, key=lambda x: x[1], reverse=True)[:top_k]
        task_experts[layer_idx] = set(e[0] for e in top_task)

    # Log expert selection
    logger.info("=== OVERLAP-AWARE ESFT EXPERT SELECTION ===")
    for layer_idx in sorted(per_layer_experts.keys())[:5]:
        shared = shared_experts.get(layer_idx, set())
        task = task_experts.get(layer_idx, set())
        logger.info(f"Layer {layer_idx}: shared={shared}, task={task}")

    # Step 4: Apply freezing
    moe_layers = get_moe_layers(model)
    total_params = 0
    trainable_params = 0
    stats = {"shared_expert_params": 0, "task_expert_params": 0, "frozen_expert_params": 0}

    for layer_idx, layer in moe_layers:
        mlp = layer.mlp
        layer_shared = shared_experts.get(layer_idx, set())
        layer_task = task_experts.get(layer_idx, set())
        layer_trainable = layer_shared | layer_task

        # Handle different expert storage formats
        experts = None
        if hasattr(mlp, "experts"):
            experts = mlp.experts
        elif hasattr(mlp, "expert_weights"):
            experts = mlp.expert_weights

        if experts is not None:
            if isinstance(experts, dict):
                expert_items = experts.items()
            elif hasattr(experts, "__iter__"):
                expert_items = enumerate(experts)
            else:
                continue

            for expert_id, expert in expert_items:
                expert_id = int(expert_id) if not isinstance(expert_id, int) else expert_id
                is_shared = expert_id in layer_shared
                is_task = expert_id in layer_task
                is_trainable = is_shared or is_task

                for param in expert.parameters():
                    param_count = param.numel()
                    total_params += param_count

                    if is_trainable:
                        param.requires_grad = True
                        trainable_params += param_count
                        if is_shared:
                            stats["shared_expert_params"] += param_count
                        else:
                            stats["task_expert_params"] += param_count
                    else:
                        param.requires_grad = False
                        stats["frozen_expert_params"] += param_count

        # Always freeze the router/gate
        gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
        if gate is not None:
            for param in gate.parameters():
                param.requires_grad = False
                total_params += param.numel()

    # Freeze non-MoE layers
    for name, param in model.named_parameters():
        if "expert" not in name.lower() and "mlp" not in name.lower():
            param.requires_grad = False

    # Report statistics
    trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0
    logger.info(
        "Overlap-aware ESFT applied",
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_pct=f"{trainable_pct:.1f}%",
        shared_expert_params=stats["shared_expert_params"],
        task_expert_params=stats["task_expert_params"],
        frozen_expert_params=stats["frozen_expert_params"],
    )

    return {
        "shared": shared_experts,
        "task": task_experts,
        "combined": {
            layer_idx: shared_experts.get(layer_idx, set()) | task_experts.get(layer_idx, set())
            for layer_idx in set(shared_experts.keys()) | set(task_experts.keys())
        },
    }


def apply_esft_partition(
    model,
    partition_id: int = 0,
    total_experts: int = 128,
    experts_per_partition: int = 32,
):
    """
    Apply partition-based ESFT for Qwen3-MoE (128 experts, no shared experts).

    This divides experts into non-overlapping partitions for distributed training.
    Each miner trains only their partition, giving 4x stronger gradients.

    Architecture (128 experts, 4 partitions):
        Partition 0: Experts [0-31]   - e.g., Math
        Partition 1: Experts [32-63]  - e.g., Code
        Partition 2: Experts [64-95]  - e.g., Vision
        Partition 3: Experts [96-127] - e.g., General

    Gradient improvement:
        Before: 8/128 = 6.25% of experts get gradients
        After:  8/32  = 25% of experts get gradients (4x improvement!)

    Args:
        model: The MoE model (Qwen3-MoE with 128 experts)
        partition_id: Which partition this miner trains (0, 1, 2, or 3)
        total_experts: Total number of experts in model (default 128)
        experts_per_partition: Experts per partition (default 32)

    Returns:
        Dict with trainable expert IDs per layer
    """
    num_partitions = total_experts // experts_per_partition
    if partition_id >= num_partitions:
        logger.warning(
            f"partition_id {partition_id} >= num_partitions {num_partitions}, "
            f"using partition_id=0"
        )
        partition_id = 0

    # Calculate expert ID range for this partition
    start_expert = partition_id * experts_per_partition
    end_expert = start_expert + experts_per_partition
    partition_experts = set(range(start_expert, end_expert))

    logger.info(
        f"=== PARTITION-BASED ESFT ===",
        partition_id=partition_id,
        expert_range=f"[{start_expert}-{end_expert-1}]",
        experts_count=len(partition_experts),
        gradient_coverage=f"{8}/{experts_per_partition} = {100*8/experts_per_partition:.1f}%",
    )

    # Apply freezing
    moe_layers = get_moe_layers(model)
    total_params = 0
    trainable_params = 0
    trainable_expert_count = 0
    frozen_expert_count = 0

    trainable_per_layer = {}

    for layer_idx, layer in moe_layers:
        mlp = layer.mlp
        trainable_per_layer[layer_idx] = set()

        # Handle different expert storage formats
        experts = None
        if hasattr(mlp, "experts"):
            experts = mlp.experts
        elif hasattr(mlp, "expert_weights"):
            experts = mlp.expert_weights

        if experts is not None:
            if isinstance(experts, dict):
                expert_items = list(experts.items())
            elif hasattr(experts, "__iter__"):
                expert_items = list(enumerate(experts))
            else:
                continue

            for expert_id, expert in expert_items:
                expert_id = int(expert_id) if not isinstance(expert_id, int) else expert_id
                is_trainable = expert_id in partition_experts

                for param in expert.parameters():
                    param_count = param.numel()
                    total_params += param_count

                    if is_trainable:
                        param.requires_grad = True
                        trainable_params += param_count
                        trainable_per_layer[layer_idx].add(expert_id)
                    else:
                        param.requires_grad = False

                if is_trainable:
                    trainable_expert_count += 1
                else:
                    frozen_expert_count += 1

        # Always freeze the router/gate (pre-trained routing is optimal)
        gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
        if gate is not None:
            for param in gate.parameters():
                param.requires_grad = False
                total_params += param.numel()

    # Freeze non-MoE layers (embeddings, attention, norms)
    for name, param in model.named_parameters():
        if "expert" not in name.lower() and "mlp" not in name.lower():
            param.requires_grad = False

    # Report statistics
    trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0
    logger.info(
        "Partition-based ESFT applied",
        partition_id=partition_id,
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_pct=f"{trainable_pct:.1f}%",
        trainable_experts=trainable_expert_count,
        frozen_experts=frozen_expert_count,
    )

    return {
        "partition_id": partition_id,
        "expert_range": (start_expert, end_expert),
        "trainable_per_layer": trainable_per_layer,
    }


def apply_esft_routing(
    model,
    expert_routing: Counter,
    top_k: int = 32,
):
    """
    Apply route-based ESFT: Train only the top-K most-routed experts.

    For Qwen3-MoE (128 experts), this selects the 32 experts that receive
    the most routing from your training data.

    Gradient improvement:
        Before: 8/128 = 6.25% gradient coverage
        After:  8/32  = 25% gradient coverage (4x improvement!)

    Args:
        model: The MoE model
        expert_routing: Counter from measure_expert_routing()
        top_k: Number of top-routed experts to train per layer (default 32)

    Returns:
        Dict with trainable expert IDs per layer
    """
    if not expert_routing:
        logger.warning("No routing data - cannot apply route-based ESFT")
        return {}

    # Parse routing data into per-layer expert rankings
    per_layer_experts = {}
    for key, count in expert_routing.items():
        if "layer_" in key and "expert_" in key:
            parts = key.split("_")
            layer_idx = int(parts[1])
            expert_id = int(parts[3])

            if layer_idx not in per_layer_experts:
                per_layer_experts[layer_idx] = []
            per_layer_experts[layer_idx].append((expert_id, count))

    # Select top-K experts per layer
    trainable_experts = {}
    for layer_idx, experts in per_layer_experts.items():
        top_experts = sorted(experts, key=lambda x: x[1], reverse=True)[:top_k]
        trainable_experts[layer_idx] = set(e[0] for e in top_experts)

    # Log expert selection for first few layers
    logger.info("=== ROUTE-BASED ESFT EXPERT SELECTION ===")
    for layer_idx in sorted(per_layer_experts.keys())[:3]:
        experts = trainable_experts.get(layer_idx, set())
        logger.info(f"Layer {layer_idx}: training {len(experts)} experts")

    # Apply freezing
    moe_layers = get_moe_layers(model)
    total_params = 0
    trainable_params = 0

    for layer_idx, layer in moe_layers:
        mlp = layer.mlp
        layer_trainable = trainable_experts.get(layer_idx, set())

        # Handle different expert storage formats
        experts = None
        if hasattr(mlp, "experts"):
            experts = mlp.experts
        elif hasattr(mlp, "expert_weights"):
            experts = mlp.expert_weights

        if experts is not None:
            if isinstance(experts, dict):
                expert_items = experts.items()
            elif hasattr(experts, "__iter__"):
                expert_items = enumerate(experts)
            else:
                continue

            for expert_id, expert in expert_items:
                expert_id = int(expert_id) if not isinstance(expert_id, int) else expert_id
                is_trainable = expert_id in layer_trainable

                for param in expert.parameters():
                    param_count = param.numel()
                    total_params += param_count

                    if is_trainable:
                        param.requires_grad = True
                        trainable_params += param_count
                    else:
                        param.requires_grad = False

        # Always freeze the router/gate
        gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
        if gate is not None:
            for param in gate.parameters():
                param.requires_grad = False
                total_params += param.numel()

    # Freeze non-MoE layers
    for name, param in model.named_parameters():
        if "expert" not in name.lower() and "mlp" not in name.lower():
            param.requires_grad = False

    # Report statistics
    trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0
    logger.info(
        "Route-based ESFT applied",
        top_k=top_k,
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_pct=f"{trainable_pct:.1f}%",
        gradient_coverage=f"8/{top_k} = {100*8/top_k:.1f}%",
    )

    return trainable_experts
