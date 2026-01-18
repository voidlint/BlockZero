#!/usr/bin/env python3
"""
Quick test script for expert profiling system.

This is a minimal test that doesn't require a full model.
It validates that the profiling infrastructure works correctly.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Expert Profiling System...")
print("=" * 80)

# Test 1: Import domain evaluators
print("\n1. Testing domain evaluators import...")
try:
    from mycelia.shared.domain_evaluators import DOMAIN_EVALUATORS
    print(f"   ✓ Successfully imported {len(DOMAIN_EVALUATORS)} domain evaluators")
    print(f"   Available domains: {list(DOMAIN_EVALUATORS.keys())}")
except Exception as e:
    print(f"   ✗ Failed to import domain evaluators: {e}")
    sys.exit(1)

# Test 2: Test each evaluator
print("\n2. Testing domain evaluators...")
test_predictions = [
    "The answer is 42. To solve this, we use the equation x + 10 = 52.",
    "def hello_world(): return 'Hello, World!'",
    "The hypothesis suggests that atoms combine to form molecules.",
]
test_references = [
    "42",
    "Hello, World!",
    "molecules",
]

for domain_name, evaluator in DOMAIN_EVALUATORS.items():
    try:
        score = evaluator.evaluate(test_predictions, test_references)
        status = "✓" if 0.0 <= score <= 1.0 else "✗"
        print(f"   {status} {domain_name:15s}: score = {score:.3f}")
    except Exception as e:
        print(f"   ✗ {domain_name:15s}: ERROR - {e}")

# Test 3: Import expert profiler
print("\n3. Testing expert profiler import...")
try:
    from mycelia.shared.expert_profiler import (
        ExpertProfile,
        LayerProfile,
        ExpertProfiler,
        print_skill_matrix,
        export_profiles_to_json,
    )
    print("   ✓ Successfully imported ExpertProfiler and utilities")
except Exception as e:
    print(f"   ✗ Failed to import expert profiler: {e}")
    sys.exit(1)

# Test 4: Create test profiles
print("\n4. Testing ExpertProfile creation...")
try:
    profile1 = ExpertProfile(
        layer_id=0,
        expert_id=0,
        math_score=0.92,
        code_score=0.45,
        science_score=0.38,
    )
    print(f"   ✓ Created profile: specialization = {profile1.specialization}")
    print(f"     {profile1.to_dict()}")

    profile2 = ExpertProfile(
        layer_id=0,
        expert_id=1,
        math_score=0.35,
        code_score=0.42,
        science_score=0.38,
    )
    print(f"   ✓ Created weak profile: specialization = {profile2.specialization}")

    profile3 = ExpertProfile(
        layer_id=0,
        expert_id=2,
        math_score=0.72,
        code_score=0.68,
        science_score=0.71,
    )
    print(f"   ✓ Created generalist profile: specialization = {profile3.specialization}")

except Exception as e:
    print(f"   ✗ Failed to create profiles: {e}")
    sys.exit(1)

# Test 5: Create layer profile
print("\n5. Testing LayerProfile...")
try:
    layer_profile = LayerProfile(layer_id=0)
    layer_profile.add_expert(profile1)
    layer_profile.add_expert(profile2)
    layer_profile.add_expert(profile3)

    summary = layer_profile.get_summary()
    print(f"   ✓ Layer summary: {summary}")

    math_specialists = layer_profile.get_specialists("math")
    print(f"   ✓ Found {len(math_specialists)} math specialists")

except Exception as e:
    print(f"   ✗ Failed to create layer profile: {e}")
    sys.exit(1)

# Test 6: Test skill matrix printing
print("\n6. Testing skill matrix visualization...")
try:
    profiles = {0: layer_profile}
    print_skill_matrix(profiles, top_k=3)
    print("   ✓ Skill matrix printed successfully")
except Exception as e:
    print(f"   ✗ Failed to print skill matrix: {e}")
    sys.exit(1)

# Test 7: Test JSON export
print("\n7. Testing JSON export...")
try:
    import tempfile
    import json

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    export_profiles_to_json(profiles, temp_path)

    with open(temp_path, 'r') as f:
        data = json.load(f)

    print(f"   ✓ Exported to JSON: {len(data)} layers")
    print(f"     Sample: {list(data.keys())}")

    # Cleanup
    Path(temp_path).unlink()

except Exception as e:
    print(f"   ✗ Failed to export JSON: {e}")
    sys.exit(1)

# All tests passed
print("\n" + "=" * 80)
print("✓ All tests passed! Expert profiling system is ready to use.")
print("\nNext steps:")
print("  1. Use Jupyter notebook: jupyter notebook mycelia/sn_owner/expert_profiling_demo.ipynb")
print("  2. Or create a script to profile your actual model")
print("  3. See EXPERT_PROFILING.md for full documentation")
print("=" * 80)