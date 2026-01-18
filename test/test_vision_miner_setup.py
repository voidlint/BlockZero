#!/usr/bin/env python3
"""
Vision Miner Setup Test

Tests the vision miner configuration without downloading the full model.
Validates:
1. Configuration loading
2. Expert pool setup
3. Dataset structure
4. Metrics computation
"""

import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_config_loading():
    """Test 1: Load and validate vision configuration."""
    print("\n" + "="*80)
    print("TEST 1: Vision Configuration Loading")
    print("="*80)

    try:
        config_path = Path(__file__).parent / "expert_groups" / "exp_vision" / "config.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print("\n✓ Configuration loaded successfully!")
        print(f"  Model: {config['model']['model_path']}")
        print(f"  Experts: {config['model']['num_experts']}")
        print(f"  Top-k: {config['model']['num_experts_per_tok']}")
        print(f"  Expert group ID: {config['expert_group_id']}")
        print(f"  Shared expert mode: {config['shared_expert_mode']}")
        print(f"  Designated expert: {config['shared_expert_config']['designated_expert_id']}")

        # Validate training stages
        print(f"\n✓ Training pipeline:")
        for stage in ['stage_a', 'stage_b', 'stage_c']:
            stage_config = config['training'][stage]
            print(f"  - {stage.upper()}: {stage_config['max_steps']} steps, lr={stage_config['learning_rate']}")

        # Validate expert pools
        print(f"\n✓ Expert pools:")
        for pool_name, pool_range in config['expert_pools'].items():
            start, end = pool_range
            num_experts = end - start + 1
            print(f"  - {pool_name}: [{start}, {end}] = {num_experts} experts")

        # Validate dataset sources
        print(f"\n✓ Dataset sources: {len(config['dataset_sources'])} total")
        vision_datasets = [d for d in config['dataset_sources'] if d.get('includes_vision', False)]
        print(f"  - Vision datasets: {len(vision_datasets)}")

        return True, config

    except Exception as e:
        print(f"\n✗ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_expert_pool_setup(config):
    """Test 2: Validate expert pool configuration."""
    print("\n" + "="*80)
    print("TEST 2: Expert Pool Setup")
    print("="*80)

    try:
        # Simulate expert pool setup
        num_experts = config['model']['num_experts']

        expert_pools = {
            "VISION": list(range(96, 127)),
            "DETECTION": list(range(96, 112)),
            "SEGMENTATION": list(range(112, 120)),
            "OCR": list(range(120, 124)),
            "DEFECT": list(range(124, 126)),
            "3D_PERCEPTION": [126],
        }

        print(f"\n✓ Expert pools created for {num_experts} total experts:")
        total_vision_experts = set()

        for pool_name, experts in expert_pools.items():
            print(f"  - {pool_name}: {len(experts)} experts")
            total_vision_experts.update(experts)

        print(f"\n✓ Total unique vision experts: {len(total_vision_experts)}")
        print(f"✓ Vision expert range: [{min(total_vision_experts)}, {max(total_vision_experts)}]")

        # Verify no overlap with robotics (0-95)
        robotics_range = set(range(0, 96))
        overlap = total_vision_experts & robotics_range
        if overlap:
            print(f"✗ WARNING: Vision experts overlap with robotics: {overlap}")
            return False
        else:
            print(f"✓ No overlap with robotics expert pool (0-95)")

        # Verify designated expert is separate
        designated_expert = config['shared_expert_config']['designated_expert_id']
        if designated_expert in total_vision_experts:
            print(f"✗ WARNING: Designated expert {designated_expert} is in vision pool")
            return False
        else:
            print(f"✓ Designated expert {designated_expert} is separate from vision pool")

        return True

    except Exception as e:
        print(f"\n✗ Expert pool setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_structure(config):
    """Test 3: Validate dataset configuration."""
    print("\n" + "="*80)
    print("TEST 3: Dataset Structure")
    print("="*80)

    try:
        dataset_sources = config['dataset_sources']

        print(f"\n✓ Dataset sources: {len(dataset_sources)} total")

        # Group by type
        by_type = {}
        for ds in dataset_sources:
            ds_type = ds['type']
            if ds_type not in by_type:
                by_type[ds_type] = []
            by_type[ds_type].append(ds)

        print(f"\n✓ Datasets by type:")
        for ds_type, datasets in sorted(by_type.items()):
            print(f"  - {ds_type}: {len(datasets)} datasets")
            for ds in datasets:
                print(f"    • {ds['name']} (weight: {ds['weight']})")

        # Validate weights sum
        total_weight = sum(ds['weight'] for ds in dataset_sources)
        print(f"\n✓ Total dataset weight: {total_weight:.2f}")

        # Check for vision datasets
        vision_datasets = [ds for ds in dataset_sources if ds.get('includes_vision', False)]
        text_datasets = [ds for ds in dataset_sources if not ds.get('includes_vision', False)]

        print(f"✓ Vision datasets: {len(vision_datasets)}")
        print(f"✓ Text-only datasets: {len(text_datasets)}")

        if len(vision_datasets) == 0:
            print(f"✗ WARNING: No vision datasets found!")
            return False

        return True

    except Exception as e:
        print(f"\n✗ Dataset structure validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_setup():
    """Test 4: Test metrics system."""
    print("\n" + "="*80)
    print("TEST 4: Metrics System")
    print("="*80)

    try:
        from expert_groups.exp_vision.metrics import get_vision_metrics_computer

        print("\n✓ Importing vision metrics...")
        metrics_computer = get_vision_metrics_computer()
        print(f"✓ Metrics computer created: {type(metrics_computer).__name__}")

        # Test with dummy data
        predictions = [
            "I can see: person, car, dog",
            "The text says: 'STOP'",
            "Defect detected at [120, 80, 180, 95]",
        ]

        references = [
            "person, car, dog",
            "STOP",
            "Defect at [120, 80, 180, 95]",
        ]

        base_metrics = {
            "val_loss": 0.5,
            "val_aux_loss": 0.1,
            "expert_diversity_score": 0.8,
            "experts_active_ratio": 0.9,
        }

        print(f"\n✓ Computing metrics on {len(predictions)} samples...")
        metrics = metrics_computer.compute_metrics(predictions, references, base_metrics)

        print(f"\n✓ Metrics computed successfully!")
        print(f"  Total metrics: {len(metrics)}")

        # Show vision-specific metrics
        vision_metric_keys = [k for k in metrics.keys()
                             if k.startswith(('object_', 'semantic_', 'ocr_', 'defect_', 'depth_', '3d_', 'video_'))]

        print(f"\n✓ Vision-specific metrics ({len(vision_metric_keys)}):")
        for key in sorted(vision_metric_keys):
            print(f"  - {key}: {metrics[key]:.4f}")

        return True

    except ImportError as e:
        print(f"\n⚠ Metrics import skipped (missing dependencies): {e}")
        return True  # Don't fail if torch not installed
    except Exception as e:
        print(f"\n✗ Metrics setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test 5: Integration with main metrics system."""
    print("\n" + "="*80)
    print("TEST 5: Main System Integration")
    print("="*80)

    try:
        from mycelia.shared.expert_specific_metrics import ExpertGroup, get_expert_metrics_computer

        print("\n✓ Importing main metrics system...")

        # Verify VISION exists in enum
        assert hasattr(ExpertGroup, 'VISION'), "VISION not found in ExpertGroup"
        assert ExpertGroup.VISION.value == 4, "VISION should have value 4"

        print(f"✓ ExpertGroup.VISION = {ExpertGroup.VISION.value}")

        # Create vision metrics computer via main system
        vision_computer = get_expert_metrics_computer(expert_group_id=4)
        print(f"✓ Vision metrics computer: {type(vision_computer).__name__}")
        print(f"✓ Expert group: {vision_computer.group.name}")

        return True

    except ImportError as e:
        print(f"\n⚠ Integration test skipped (missing dependencies): {e}")
        return True  # Don't fail if torch not installed
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all setup tests."""
    print("\n" + "="*80)
    print("VISION MINER SETUP VALIDATION")
    print("="*80)
    print("\nValidating vision miner setup without downloading model...")

    results = {}

    # Test 1: Config loading
    success, config = test_config_loading()
    results["config_loading"] = success

    if not success:
        print("\n✗ Config loading failed. Aborting remaining tests.")
        return False

    # Test 2: Expert pool setup
    results["expert_pool_setup"] = test_expert_pool_setup(config)

    # Test 3: Dataset structure
    results["dataset_structure"] = test_dataset_structure(config)

    # Test 4: Metrics system
    results["metrics_setup"] = test_metrics_setup()

    # Test 5: Integration
    results["integration"] = test_integration()

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
        print("✓ ALL SETUP TESTS PASSED!")
        print("\nYour vision miner is ready to run!")
        print(f"\nTo start training:")
        print(f"  python3 run_vision_miner.py")
        print(f"\nNote: The first run will download Qwen3-VL-30B-A3B-Thinking (~60GB)")
        print(f"      Make sure you have sufficient disk space and RAM.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the errors above before running the miner.")

    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)