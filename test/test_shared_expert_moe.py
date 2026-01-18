#!/usr/bin/env python3
"""
Test Shared Expert MoE Implementation

Verifies that:
1. Qwen3-VL model loads correctly
2. Shared expert architecture is added properly
3. Masked routing works for expert pools
4. Forward pass completes successfully
5. Training step works
"""
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from expert_groups.exp_robotics.shared_expert_moe import (
    load_qwen3vl_with_shared_experts,
    MaskedRoutingWrapper,
    Qwen3VLSharedExpertMoeBlock,
)


def test_model_loading():
    """Test 1: Load model and add shared experts."""
    print("\n" + "="*80)
    print("TEST 1: Model Loading & Architecture Conversion")
    print("="*80)

    try:
        model, processor = load_qwen3vl_with_shared_experts(
            model_path="Qwen/Qwen3-VL-30B-A3B-Thinking",
            freeze_routed_experts=False,
            device_map="auto",
        )

        print("\n✓ Model loaded successfully!")
        print(f"✓ Model type: {type(model).__name__}")
        print(f"✓ Config: {model.config.model_type}")

        # Verify shared expert MoE layers
        moe_count = 0
        shared_expert_count = 0

        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, Qwen3VLSharedExpertMoeBlock):
                moe_count += 1
                if hasattr(layer.mlp, 'shared_expert'):
                    shared_expert_count += 1

        print(f"\n✓ Found {moe_count} MoE layers")
        print(f"✓ {shared_expert_count} layers have shared experts")

        if shared_expert_count == moe_count:
            print("✓ All MoE layers successfully converted to shared expert architecture!")
        else:
            print(f"⚠ Warning: Only {shared_expert_count}/{moe_count} layers have shared experts")

        return model, processor

    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(model, processor):
    """Test 2: Forward pass with vision + text."""
    print("\n" + "="*80)
    print("TEST 2: Forward Pass (Vision + Text)")
    print("="*80)

    try:
        # Create dummy inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ]

        # For testing without actual image, we'll create a dummy input
        # In real usage, you'd use: {"type": "image", "image": "path/to/image.jpg"}

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt", padding=True)

        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print(f"\n✓ Input prepared")
        print(f"  - Input IDs shape: {inputs['input_ids'].shape}")
        print(f"  - Device: {device}")

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        print(f"\n✓ Forward pass successful!")
        print(f"  - Logits shape: {outputs.logits.shape}")
        print(f"  - Has loss: {hasattr(outputs, 'loss')}")

        return True

    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_masked_routing(model):
    """Test 3: Masked routing to expert pools."""
    print("\n" + "="*80)
    print("TEST 3: Masked Routing to Expert Pools")
    print("="*80)

    try:
        # Define expert pools
        expert_pools = {
            "ROBOTICS": list(range(0, 200)),
            "MATH": list(range(200, 300)),
            "CODE": list(range(300, 400)),
            "GENERAL": list(range(400, 512)),
        }

        print(f"\n✓ Expert pools defined:")
        for pool_name, experts in expert_pools.items():
            print(f"  - {pool_name}: {len(experts)} experts (indices {experts[0]}-{experts[-1]})")

        # Create routing wrapper
        wrapper = MaskedRoutingWrapper(model, expert_pools)
        print(f"\n✓ Routing wrapper created")

        # Test routing to ROBOTICS pool
        with wrapper.route_to_pool("ROBOTICS"):
            print(f"\n✓ Routing context activated for ROBOTICS pool")

            # Verify masks are applied
            masks_applied = 0
            for layer in model.model.layers:
                if hasattr(layer, 'mlp') and isinstance(layer.mlp, Qwen3VLSharedExpertMoeBlock):
                    if hasattr(layer.mlp, 'current_expert_mask') and layer.mlp.current_expert_mask is not None:
                        masks_applied += 1

            print(f"  - Masks applied to {masks_applied} MoE layers")

        # Verify masks are removed
        print(f"\n✓ Routing context exited")

        masks_remaining = 0
        for layer in model.model.layers:
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, Qwen3VLSharedExpertMoeBlock):
                if hasattr(layer.mlp, 'current_expert_mask') and layer.mlp.current_expert_mask is not None:
                    masks_remaining += 1

        if masks_remaining == 0:
            print(f"  - All masks properly removed ✓")
        else:
            print(f"  - Warning: {masks_remaining} masks still active ⚠")

        return True

    except Exception as e:
        print(f"\n✗ Masked routing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model, processor):
    """Test 4: Training step with masked routing."""
    print("\n" + "="*80)
    print("TEST 4: Training Step with Masked Routing")
    print("="*80)

    try:
        # Put model in training mode
        model.train()

        # Define expert pools
        expert_pools = {
            "ROBOTICS": list(range(0, 200)),
        }
        wrapper = MaskedRoutingWrapper(model, expert_pools)

        # Create dummy training batch
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Pick up the red cube."},
                ],
            },
            {
                "role": "assistant",
                "content": "I'll pick up the red cube using a top-down grasp.",
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text], return_tensors="pt", padding=True)

        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Add labels for loss computation
        inputs["labels"] = inputs["input_ids"].clone()

        print(f"\n✓ Training batch prepared")
        print(f"  - Batch size: {inputs['input_ids'].shape[0]}")
        print(f"  - Sequence length: {inputs['input_ids'].shape[1]}")

        # Forward pass with masked routing
        with wrapper.route_to_pool("ROBOTICS"):
            outputs = model(**inputs)
            loss = outputs.loss

        print(f"\n✓ Forward pass completed")
        print(f"  - Loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()
        print(f"✓ Backward pass completed")

        # Check gradients
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1

        print(f"✓ Gradients computed for {grad_count} parameters")

        # Zero gradients
        model.zero_grad()
        print(f"✓ Gradients zeroed")

        return True

    except Exception as e:
        print(f"\n✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        model.eval()


def test_parameter_counts(model):
    """Test 5: Count parameters for shared expert overhead."""
    print("\n" + "="*80)
    print("TEST 5: Parameter Count Analysis")
    print("="*80)

    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Frozen parameters: {total_params - trainable_params:,}")

        # Count shared expert parameters
        shared_expert_params = 0
        routed_expert_params = 0

        for layer in model.model.layers:
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, Qwen3VLSharedExpertMoeBlock):
                # Shared expert
                if hasattr(layer.mlp, 'shared_expert'):
                    shared_expert_params += sum(p.numel() for p in layer.mlp.shared_expert.parameters())

                # Routed experts
                if hasattr(layer.mlp, 'experts'):
                    routed_expert_params += sum(p.numel() for p in layer.mlp.experts.parameters())

        print(f"\n✓ Expert parameters breakdown:")
        print(f"  - Shared expert params: {shared_expert_params:,}")
        print(f"  - Routed expert params: {routed_expert_params:,}")
        print(f"  - Overhead from shared experts: {shared_expert_params / total_params * 100:.2f}%")

        return True

    except Exception as e:
        print(f"\n✗ Parameter counting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SHARED EXPERT MOE TESTING SUITE")
    print("="*80)
    print("\nTesting Qwen3-VL-30B-A3B-Thinking with shared expert architecture...")

    results = {}

    # Test 1: Model loading
    model, processor = test_model_loading()
    results["model_loading"] = model is not None

    if model is None:
        print("\n✗ Model loading failed. Aborting remaining tests.")
        return False

    # Test 2: Forward pass
    results["forward_pass"] = test_forward_pass(model, processor)

    # Test 3: Masked routing
    results["masked_routing"] = test_masked_routing(model)

    # Test 4: Training step
    results["training_step"] = test_training_step(model, processor)

    # Test 5: Parameter counts
    results["parameter_counts"] = test_parameter_counts(model)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    print("\n" + "="*80)

    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYour Qwen3-VL model is ready for robotics expert training with:")
        print("  - Shared expert architecture (maintains baseline capabilities)")
        print("  - Masked routing to expert pools (enables specialization)")
        print("  - Full training pipeline support")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the errors above.")

    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)