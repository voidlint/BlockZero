#!/usr/bin/env python3
"""
Real Expert Profiling Script

Profiles actual MoE model experts across multiple domains.
This script loads your trained model and evaluates individual experts.

Usage:
    # Quick test
    python run_expert_profiling.py

    # Profile specific layer
    LAYER=0 NUM_SAMPLES=20 python run_expert_profiling.py

    # Profile specific expert
    LAYER=0 EXPERT=5 NUM_SAMPLES=50 python run_expert_profiling.py

    # Multiple domains
    DOMAINS="math,code,science,reasoning" NUM_SAMPLES=30 python run_expert_profiling.py
"""
import sys
import os
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from mycelia.shared.config import ValidatorConfig
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.expert_profiler import (
    ExpertProfiler,
    print_skill_matrix,
    export_profiles_to_json,
)

def main():
    # Use environment variables for configuration (to avoid argparse conflicts)
    model_path = os.getenv("MODEL_PATH", None)
    layer = int(os.getenv("LAYER")) if os.getenv("LAYER") else None
    expert = int(os.getenv("EXPERT")) if os.getenv("EXPERT") else None
    num_samples = int(os.getenv("NUM_SAMPLES", "10"))
    domains_str = os.getenv("DOMAINS", "math,code,science")
    output_file = os.getenv("OUTPUT", "expert_profiles.json")
    device_str = os.getenv("DEVICE", None)
    quick_test = os.getenv("QUICK_TEST", "").lower() in ("1", "true", "yes")

    print("="*80)
    print("EXPERT PROFILING SYSTEM")
    print("="*80)

    # Configuration
    config = ValidatorConfig()

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

    # Quick test mode
    if quick_test:
        print("\n⚡ QUICK TEST MODE")
        num_samples = 5
        domains_str = "math"
        print(f"   - Testing 1 expert")
        print(f"   - {num_samples} samples per domain")
        print(f"   - Domains: {domains_str}")

    # Load model
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
        print("\n💡 TIP: Make sure you have a trained model at the path specified.")
        print(f"   Current path: {model_path}")
        return 1

    # Load tokenizer
    print(f"\n📝 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"   ✓ Tokenizer loaded")
    except Exception as e:
        print(f"   ✗ Failed to load tokenizer: {e}")
        return 1

    # Initialize profiler
    print(f"\n🔍 Initializing profiler...")
    try:
        profiler = ExpertProfiler(
            model=model,
            tokenizer=tokenizer,
            device=device,
            num_samples_per_domain=num_samples,
        )
        print(f"   ✓ Detected {len(profiler.moe_layers)} MoE layers: {profiler.moe_layers}")
    except Exception as e:
        print(f"   ✗ Failed to initialize profiler: {e}")
        return 1

    if not profiler.moe_layers:
        print("\n   ⚠️  No MoE layers detected in this model!")
        print("   This profiler only works with Mixture-of-Experts models.")
        return 1

    # Prepare dataloaders for requested domains
    print(f"\n📊 Preparing dataloaders for domains: {domains_str}")
    domains = [d.strip() for d in domains_str.split(",")]
    domain_dataloaders = {}

    for domain in domains:
        try:
            # Map domain names to expert group IDs
            domain_to_group = {
                "math": 0,
                "agentic": 1,
                "planning": 2,
                "code": 0,  # Use math data as proxy
                "science": 0,  # Use math data as proxy
                "reasoning": 2,  # Use planning data as proxy
            }

            group_id = domain_to_group.get(domain, 0)

            # Create dataloader config
            domain_config = ValidatorConfig()
            domain_config.task.data.batch_size = 1
            domain_config.task.expert_group_id = group_id

            dataloader = get_dataloader(
                domain_config,
                rank=0,
                world_size=1,
                tokenizer=tokenizer,
            )

            domain_dataloaders[domain] = dataloader
            print(f"   ✓ {domain}: using expert group {group_id} data")

        except Exception as e:
            print(f"   ⚠️  {domain}: Failed to create dataloader ({e}), skipping")

    if not domain_dataloaders:
        print("\n   ✗ No valid dataloaders created. Cannot proceed.")
        return 1

    # Profile experts
    print("\n"+"="*80)
    print("PROFILING EXPERTS")
    print("="*80)

    layer_profiles = {}

    # Determine which layers to profile
    if layer is not None:
        if layer in profiler.moe_layers:
            layers_to_profile = [layer]
        else:
            print(f"\n✗ Layer {layer} is not a MoE layer!")
            print(f"   Available MoE layers: {profiler.moe_layers}")
            return 1
    else:
        layers_to_profile = profiler.moe_layers

    # Profile each layer
    for layer_id in layers_to_profile:
        print(f"\n{'─'*80}")
        print(f"LAYER {layer_id}")
        print(f"{'─'*80}")

        num_experts_in_layer = profiler._get_expert_count(layer_id)
        print(f"Total experts in layer: {num_experts_in_layer}")

        # Determine which experts to profile
        if expert is not None:
            if expert < num_experts_in_layer:
                experts_to_profile = [expert]
            else:
                print(f"✗ Expert {expert} doesn't exist (layer has {num_experts_in_layer} experts)")
                continue
        elif quick_test:
            experts_to_profile = [0]  # Just first expert in quick test
        else:
            experts_to_profile = range(num_experts_in_layer)

        print(f"Profiling {len(experts_to_profile)} expert(s)...\n")

        # Profile each expert
        from mycelia.shared.expert_profiler import LayerProfile
        layer_profile = LayerProfile(layer_id=layer_id)

        for expert_id in experts_to_profile:
            print(f"   Expert {expert_id}:")

            try:
                expert_profile = profiler.profile_expert(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    domain_dataloaders=domain_dataloaders,
                )

                # Show results
                profile_dict = expert_profile.to_dict()
                print(f"      Specialization: {profile_dict['specialization']}")

                for domain in domains:
                    score_key = f"{domain}_score"
                    if score_key in profile_dict:
                        score = profile_dict[score_key]
                        bar = "█" * int(score * 20)
                        print(f"      {domain:12s}: {score:.3f} {bar}")

                print(f"      Avg Loss: {profile_dict['avg_loss']:.3f}")

                layer_profile.add_expert(expert_profile)

            except Exception as e:
                print(f"      ✗ Error: {e}")
                continue

        layer_profiles[layer_id] = layer_profile

    # Print summary
    if layer_profiles:
        print("\n"+"="*80)
        print("SUMMARY")
        print("="*80)

        print_skill_matrix(layer_profiles, top_k=5)

        # Export to JSON
        try:
            export_profiles_to_json(layer_profiles, output_file)
            print(f"\n✓ Results exported to: {output_file}")
        except Exception as e:
            print(f"\n⚠️  Failed to export JSON: {e}")

    print("\n"+"="*80)
    print("PROFILING COMPLETE")
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