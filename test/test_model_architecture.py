#!/usr/bin/env python3
"""
Test Qwen3-VL model architecture to understand layer structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("QWEN3-VL MODEL ARCHITECTURE INSPECTION")
print("="*80)

# Test 1: Check if model config loads
print("\n[1/5] Loading model configuration...")
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Thinking",
        trust_remote_code=True
    )
    print(f"✓ Config loaded: {type(config).__name__}")
    print(f"  - Experts: {getattr(config, 'num_experts', 'N/A')}")
    print(f"  - Experts per token: {getattr(config, 'num_experts_per_tok', 'N/A')}")
    print(f"  - Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Check MoE classes are available
print("\n[2/5] Checking MoE classes...")
try:
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeTextSparseMoeBlock,
        Qwen3VLMoeTextMLP,
        Qwen3VLMoeTextDecoderLayer,
    )
    print("✓ MoE classes imported successfully")
    print(f"  - Qwen3VLMoeTextSparseMoeBlock: {Qwen3VLMoeTextSparseMoeBlock}")
    print(f"  - Qwen3VLMoeTextMLP: {Qwen3VLMoeTextMLP}")
    print(f"  - Qwen3VLMoeTextDecoderLayer: {Qwen3VLMoeTextDecoderLayer}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check SharedExpertMoE imports
print("\n[3/5] Testing SharedExpertMoE module...")
try:
    from expert_groups.exp_robotics.shared_expert_moe import (
        Qwen3VLSharedExpertMoeBlock,
        convert_qwen3vl_to_shared_expert_moe,
    )
    print("✓ SharedExpertMoE module imported successfully")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Inspect config text attributes
print("\n[4/5] Inspecting text model config...")
try:
    text_config = config.text_config
    print(f"✓ Text config: {type(text_config).__name__}")
    print(f"  - num_hidden_layers: {getattr(text_config, 'num_hidden_layers', 'N/A')}")
    print(f"  - num_experts: {getattr(text_config, 'num_experts', 'N/A')}")
    print(f"  - num_experts_per_tok: {getattr(text_config, 'num_experts_per_tok', 'N/A')}")
    print(f"  - hidden_size: {getattr(text_config, 'hidden_size', 'N/A')}")
    print(f"  - moe_intermediate_size: {getattr(text_config, 'moe_intermediate_size', 'N/A')}")
except Exception as e:
    print(f"✗ Warning: {e}")

# Test 5: Create minimal model block for testing
print("\n[5/5] Testing SharedExpertMoeBlock initialization...")
try:
    import torch
    from expert_groups.exp_robotics.shared_expert_moe import Qwen3VLSharedExpertMoeBlock

    # Use text config for initialization
    test_config = config.text_config

    # Create a test MoE block
    test_block = Qwen3VLSharedExpertMoeBlock(
        config=test_config,
        original_moe_block=None,  # Initialize from scratch
        shared_expert_mode="designated",
        designated_expert_id=127,
        bias_strength=10.0,
    )

    print(f"✓ SharedExpertMoeBlock created successfully")
    print(f"  - Num experts: {test_block.num_experts}")
    print(f"  - Top-k: {test_block.top_k}")
    print(f"  - Designated expert: {test_block.designated_expert_id}")
    print(f"  - Hidden dim: {test_block.hidden_dim}")

    # Test forward pass with dummy data
    batch_size, seq_len, hidden_dim = 2, 4, test_block.hidden_dim
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim)

    print(f"\n  Testing forward pass...")
    output, router_logits = test_block(dummy_input)
    print(f"  ✓ Forward pass successful")
    print(f"    - Output shape: {output.shape}")
    print(f"    - Router logits shape: {router_logits.shape}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓ ALL TESTS PASSED")
print("="*80)
print("\nThe SharedExpertMoE implementation is compatible with Qwen3-VL architecture!")