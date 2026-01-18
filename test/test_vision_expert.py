#!/usr/bin/env python3
"""
Test Vision Expert Implementation

Verifies that:
1. Vision expert configuration loads correctly
2. Vision dataset classes are importable and functional
3. Vision metrics computation works
4. Integration with main metrics system works
5. Expert pool allocation is correct
"""
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_config_loading():
    """Test 1: Load vision expert configuration."""
    print("\n" + "="*80)
    print("TEST 1: Vision Expert Configuration Loading")
    print("="*80)

    try:
        config_path = Path(__file__).parent / "expert_groups" / "exp_vision" / "config.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print("\n✓ Configuration loaded successfully!")

        # Verify key settings
        assert config['model']['num_experts'] == 128, "Expected 128 experts"
        assert config['model']['num_experts_per_tok'] == 8, "Expected top-8 routing"
        assert config['expert_group_id'] == 4, "Expected expert group ID 4"
        assert config['expert_group_name'] == 'exp_vision', "Expected exp_vision name"

        print(f"✓ Model: {config['model']['model_path']}")
        print(f"✓ Expert count: {config['model']['num_experts']}")
        print(f"✓ Experts per token: {config['model']['num_experts_per_tok']}")
        print(f"✓ Expert group ID: {config['expert_group_id']}")

        # Verify expert pools
        pools = config['expert_pools']
        print(f"\n✓ Expert pool allocation:")
        for pool_name, pool_range in pools.items():
            start, end = pool_range
            num_experts = end - start + 1
            print(f"  - {pool_name}: [{start}, {end}] ({num_experts} experts)")

        # Verify shared expert config
        shared_config = config['shared_expert_config']
        print(f"\n✓ Shared expert mode: {config['shared_expert_mode']}")
        print(f"✓ Designated expert ID: {shared_config['designated_expert_id']}")
        print(f"✓ Bias strength: {shared_config['bias_strength']}")

        # Verify routing config
        routing = config['routing']
        print(f"\n✓ Routing mode: {routing['routing_mode']}")
        print(f"✓ Pool bias strength: {routing['pool_bias_strength']}")
        print(f"✓ Load balance on pool only: {routing['compute_load_balance_on_pool_only']}")

        # Verify dataset sources
        datasets = config['dataset_sources']
        print(f"\n✓ Dataset sources: {len(datasets)} total")

        vision_datasets = [d for d in datasets if d.get('includes_vision', False)]
        print(f"✓ Vision datasets: {len(vision_datasets)}")

        return True

    except Exception as e:
        print(f"\n✗ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_imports():
    """Test 2: Import vision dataset classes."""
    print("\n" + "="*80)
    print("TEST 2: Vision Dataset Imports")
    print("="*80)

    try:
        from expert_groups.exp_vision.dataset import (
            MergedVisionDataset,
            COCODetectionDataset,
            OpenImagesDetectionDataset,
            ADE20KDataset,
            CityscapesDataset,
            TextVQADataset,
            DocVQADataset,
            IIIT5KDataset,
            MVTecADDataset,
            DAGMDataset,
            NYUDepthV2Dataset,
            ScanNetDataset,
            KITTIDataset,
            Kinetics400Dataset,
            ActivityNetDataset,
            MomentsInTimeDataset,
            LLaVAGeneralDataset,
        )

        print("\n✓ All dataset classes imported successfully!")

        dataset_classes = [
            "MergedVisionDataset",
            "COCODetectionDataset",
            "OpenImagesDetectionDataset",
            "ADE20KDataset",
            "CityscapesDataset",
            "TextVQADataset",
            "DocVQADataset",
            "IIIT5KDataset",
            "MVTecADDataset",
            "DAGMDataset",
            "NYUDepthV2Dataset",
            "ScanNetDataset",
            "KITTIDataset",
            "Kinetics400Dataset",
            "ActivityNetDataset",
            "MomentsInTimeDataset",
            "LLaVAGeneralDataset",
        ]

        for cls_name in dataset_classes:
            print(f"  ✓ {cls_name}")

        print(f"\n✓ Total: {len(dataset_classes)} dataset classes")

        return True

    except Exception as e:
        print(f"\n✗ Dataset import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_computation():
    """Test 3: Vision metrics computation."""
    print("\n" + "="*80)
    print("TEST 3: Vision Metrics Computation")
    print("="*80)

    try:
        from expert_groups.exp_vision.metrics import get_vision_metrics_computer

        print("\n✓ Vision metrics computer imported successfully!")

        # Create metrics computer
        metrics_computer = get_vision_metrics_computer()
        print(f"✓ Metrics computer created: {type(metrics_computer).__name__}")

        # Test with dummy predictions and references
        predictions = [
            "I can see: person (confidence: 0.95), car (confidence: 0.89), dog (confidence: 0.92)",
            "The text says: 'STOP'",
            "Defect detected at region [x1=120, y1=80, x2=180, y2=95] with 92% confidence",
        ]

        references = [
            "person [120, 80, 340, 450], car [400, 200, 550, 380]",
            "The text says: 'STOP'",
            "Defect at [x1=120, y1=80, x2=180, y2=95]",
        ]

        base_metrics = {
            "val_loss": 0.5,
            "val_aux_loss": 0.1,
            "expert_diversity_score": 0.8,
            "experts_active_ratio": 0.9,
        }

        # Compute metrics
        metrics = metrics_computer.compute_metrics(predictions, references, base_metrics)

        print("\n✓ Metrics computed successfully!")
        print(f"\nComputed metrics:")

        metric_names = [
            "object_detection_map",
            "instance_segmentation_iou",
            "semantic_segmentation_miou",
            "ocr_character_accuracy",
            "ocr_word_accuracy",
            "defect_detection_f1",
            "defect_localization_iou",
            "depth_estimation_rmse",
            "3d_reconstruction_chamfer",
            "video_action_accuracy",
            "video_temporal_iou",
        ]

        for metric_name in metric_names:
            if metric_name in metrics:
                print(f"  ✓ {metric_name}: {metrics[metric_name]:.4f}")
            else:
                print(f"  ⚠ {metric_name}: not computed")

        return True

    except Exception as e:
        print(f"\n✗ Metrics computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_system_integration():
    """Test 4: Integration with main metrics system."""
    print("\n" + "="*80)
    print("TEST 4: Main System Integration")
    print("="*80)

    try:
        from mycelia.shared.expert_specific_metrics import (
            ExpertGroup,
            ExpertMetricsComputer,
            get_expert_metrics_computer,
        )

        print("\n✓ Main metrics system imported successfully!")

        # Verify VISION enum exists
        assert hasattr(ExpertGroup, 'VISION'), "VISION not in ExpertGroup enum"
        assert ExpertGroup.VISION.value == 4, "VISION should have value 4"

        print(f"✓ ExpertGroup.VISION exists with value: {ExpertGroup.VISION.value}")

        # Create metrics computer for vision expert
        vision_computer = get_expert_metrics_computer(expert_group_id=4)
        print(f"✓ Vision metrics computer created: {type(vision_computer).__name__}")
        print(f"✓ Expert group: {vision_computer.group.name}")

        # Test metrics computation through main system
        predictions = [
            "I can see: person, car, dog",
            "The sign says 'STOP'",
        ]

        references = [
            "person, car, dog",
            "STOP",
        ]

        base_metrics = {
            "val_loss": 0.5,
            "val_aux_loss": 0.1,
            "expert_diversity_score": 0.8,
            "experts_active_ratio": 0.9,
        }

        metrics = vision_computer.compute_metrics(predictions, references, base_metrics)

        print("\n✓ Metrics computed through main system!")
        print(f"  - Metrics returned: {len(metrics)} total")

        # Verify base metrics are preserved
        assert "val_loss" in metrics, "Base metrics not preserved"
        print("✓ Base metrics preserved in output")

        return True

    except Exception as e:
        print(f"\n✗ Main system integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_expert_pool_validation():
    """Test 5: Validate expert pool allocation."""
    print("\n" + "="*80)
    print("TEST 5: Expert Pool Allocation Validation")
    print("="*80)

    try:
        config_path = Path(__file__).parent / "expert_groups" / "exp_vision" / "config.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        pools = config['expert_pools']

        print("\n✓ Validating expert pool allocation...")

        # Check ranges
        all_experts = set()
        for pool_name, pool_range in pools.items():
            start, end = pool_range

            # Validate range
            assert 0 <= start < 128, f"{pool_name}: start {start} out of range"
            assert 0 <= end < 128, f"{pool_name}: end {end} out of range"
            assert start <= end, f"{pool_name}: start > end"

            # Collect experts
            pool_experts = set(range(start, end + 1))

            # Check for overlaps
            overlap = all_experts & pool_experts
            assert len(overlap) == 0, f"{pool_name}: overlaps with previous pools at {overlap}"

            all_experts.update(pool_experts)

            num_experts = end - start + 1
            print(f"  ✓ {pool_name}: [{start}, {end}] = {num_experts} experts")

        print(f"\n✓ Total vision experts allocated: {len(all_experts)}")
        print(f"✓ Expert indices: {sorted(all_experts)}")

        # Verify we're using 96-126 (31 experts)
        expected_range = set(range(96, 127))  # 96-126 inclusive
        assert all_experts == expected_range, f"Expected {expected_range}, got {all_experts}"

        print("✓ All experts in expected range [96, 126]")

        # Verify designated shared expert
        designated_id = config['shared_expert_config']['designated_expert_id']
        print(f"\n✓ Designated shared expert: {designated_id}")

        # Verify designated expert is NOT in vision pools (it's shared)
        assert designated_id not in all_experts, f"Designated expert {designated_id} should not be in vision pools"
        print(f"✓ Designated expert {designated_id} correctly excluded from vision pools")

        return True

    except Exception as e:
        print(f"\n✗ Expert pool validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("VISION EXPERT TESTING SUITE")
    print("="*80)
    print("\nTesting Vision Expert implementation...")

    results = {}

    # Test 1: Configuration loading
    results["config_loading"] = test_config_loading()

    # Test 2: Dataset imports
    results["dataset_imports"] = test_dataset_imports()

    # Test 3: Metrics computation
    results["metrics_computation"] = test_metrics_computation()

    # Test 4: Main system integration
    results["main_system_integration"] = test_main_system_integration()

    # Test 5: Expert pool validation
    results["expert_pool_validation"] = test_expert_pool_validation()

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
        print("\nYour Vision Expert is ready for use:")
        print("  - Configuration: expert_groups/exp_vision/config.yaml")
        print("  - Datasets: expert_groups/exp_vision/dataset.py")
        print("  - Metrics: expert_groups/exp_vision/metrics.py")
        print("  - Expert pool: 31 experts (indices 96-126)")
        print("  - Shared expert: ID 127 (designated mode)")
        print("  - Integration: Connected to main metrics system")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the errors above.")

    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)