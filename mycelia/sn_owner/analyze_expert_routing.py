#!/usr/bin/env python3
"""
Analyze Expert Routing

Shows which experts are selected by the router for specific domains.
Much faster than profiling - just observes natural routing behavior.

Usage:
    # Analyze math domain
    python analyze_expert_routing.py

    # Analyze specific domain with more samples
    DOMAIN=math NUM_SAMPLES=100 python analyze_expert_routing.py

    # Analyze multiple domains
    DOMAIN=math,code,science NUM_SAMPLES=50 python analyze_expert_routing.py
"""
import sys
import os
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from mycelia.shared.config import ValidatorConfig
from mycelia.shared.dataloader import get_dataloader

def analyze_routing(
    model,
    tokenizer,
    dataloader,
    num_samples: int = 50,
    device: torch.device = torch.device("cpu")
):
    """
    Analyze which experts are selected for given data.

    Returns:
        Dict mapping layer_id -> expert_id -> selection_count
    """
    model.eval()

    # Track expert selections per layer
    expert_selections = defaultdict(lambda: defaultdict(int))

    # Hook to capture router outputs
    router_outputs = {}

    def make_hook(layer_id):
        def hook(module, input, output):
            # Output format: (hidden_states, router_logits)
            if isinstance(output, tuple) and len(output) >= 2:
                router_logits = output[1]  # [batch*seq_len, num_experts]

                # Get top-k selected experts
                if hasattr(module.gate, 'top_k'):
                    top_k = module.gate.top_k
                else:
                    top_k = 2  # Default

                # Get selected experts
                _, selected = torch.topk(router_logits, top_k, dim=-1)
                router_outputs[layer_id] = selected.cpu()
        return hook

    # Register hooks on MoE layers
    hooks = []
    moe_layers = []

    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            moe_layers.append(layer_idx)
            hook = layer.mlp.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

    print(f"Found {len(moe_layers)} MoE layers: {moe_layers[:5]}{'...' if len(moe_layers) > 5 else ''}")

    # Process samples
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break

            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{num_samples}...", end='\r')

            # Move to device
            device_batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            try:
                outputs = model(**device_batch)

                # Count expert selections
                for layer_id, selected_experts in router_outputs.items():
                    # selected_experts: [batch*seq_len, top_k]
                    for expert_id in selected_experts.flatten().tolist():
                        expert_selections[layer_id][expert_id] += 1

                router_outputs.clear()

            except Exception as e:
                print(f"\nError on batch {batch_idx}: {e}")
                continue

    print(f"\nProcessed {min(batch_idx + 1, num_samples)} batches")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return expert_selections, moe_layers


def print_top_experts(expert_selections: Dict, moe_layers: List[int], top_k: int = 3):
    """Print top selected experts per layer."""
    print("\n" + "="*80)
    print("TOP SELECTED EXPERTS PER LAYER")
    print("="*80)

    for layer_id in sorted(expert_selections.keys()):
        selections = expert_selections[layer_id]
        total_selections = sum(selections.values())

        # Sort by selection count
        top_experts = sorted(selections.items(), key=lambda x: x[1], reverse=True)[:top_k]

        print(f"\nLayer {layer_id}:")
        print(f"  Total selections: {total_selections}")
        print(f"  Top {top_k} experts:")

        for rank, (expert_id, count) in enumerate(top_experts, 1):
            percentage = (count / total_selections * 100) if total_selections > 0 else 0
            bar = "█" * int(percentage / 5)  # Scale to ~20 chars max
            print(f"    {rank}. Expert {expert_id:3d}: {count:6d} selections ({percentage:5.1f}%) {bar}")


def main():
    # Configuration from environment
    domain = os.getenv("DOMAIN", "math")
    num_samples = int(os.getenv("NUM_SAMPLES", "50"))
    device_str = os.getenv("DEVICE", None)
    model_path = os.getenv("MODEL_PATH", None)

    print("="*80)
    print("EXPERT ROUTING ANALYSIS")
    print("="*80)

    # Device selection
    if device_str:
        device = torch.device(device_str)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"\n📱 Device: {device}")

    # Configuration
    config = ValidatorConfig()
    if model_path is None:
        model_path = config.model.model_path

    print(f"\n📦 Loading model from: {model_path}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )

        if device.type != "cuda":
            model = model.to(device)

        model.eval()
        print(f"   ✓ Model loaded: {model.__class__.__name__}")

    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return 1

    # Load tokenizer
    print(f"\n📝 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"   ✓ Tokenizer loaded")
    except Exception as e:
        print(f"   ✗ Failed to load tokenizer: {e}")
        return 1

    # Map domain to expert group
    domain_to_group = {
        "math": 0,
        "agentic": 1,
        "planning": 2,
        "code": 0,
        "science": 0,
        "reasoning": 2,
    }

    domains = [d.strip() for d in domain.split(",")]

    for domain_name in domains:
        group_id = domain_to_group.get(domain_name, 0)

        print(f"\n📊 Analyzing {domain_name} domain (expert group {group_id})")
        print(f"   Samples: {num_samples}")

        # Create dataloader
        try:
            domain_config = ValidatorConfig()
            domain_config.task.data.batch_size = 1
            domain_config.task.expert_group_id = group_id

            dataloader = get_dataloader(
                domain_config,
                rank=0,
                world_size=1,
                tokenizer=tokenizer,
            )
            print(f"   ✓ Dataloader created")

        except Exception as e:
            print(f"   ✗ Failed to create dataloader: {e}")
            continue

        # Analyze routing
        print(f"\n🔍 Analyzing expert routing...")
        expert_selections, moe_layers = analyze_routing(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            num_samples=num_samples,
            device=device,
        )

        # Print results
        print_top_experts(expert_selections, moe_layers, top_k=5)

        # Export to JSON
        import json
        output_file = f"expert_routing_{domain_name}.json"

        export_data = {
            "domain": domain_name,
            "num_samples": num_samples,
            "moe_layers": moe_layers,
            "expert_selections": {
                str(layer_id): {
                    str(expert_id): count
                    for expert_id, count in selections.items()
                }
                for layer_id, selections in expert_selections.items()
            }
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\n✓ Results exported to: {output_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)