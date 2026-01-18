# ESFT Security: Preventing Miner Bypass

## Problem

**Original Risk:** Miners could run ESFT (Expert-Specialized Fine-Tuning) locally to select their own experts, bypassing the centralized assignment system.

```python
# SECURITY RISK: Miner could do this
from mycelia.shared.esft_selector import run_esft_expert_selection

# Select "easy" experts locally
selected_experts = run_esft_expert_selection(model, dataloader, config)
# Train only these self-selected experts → unfair advantage
```

This would allow miners to:
1. Choose "easier" experts that are simpler to train
2. Avoid difficult/complex expert assignments
3. Game the reward system
4. Bypass SN owner's fair distribution

## Solution: Three-Layer Defense

### Layer 1: Runtime Detection

**File:** `mycelia/shared/esft_selector.py`

```python
def run_esft_expert_selection(..., allow_miner_execution: bool = False):
    """SECURITY: Restricted to SN owner only."""
    
    # Check if we're in a miner context
    is_miner_context = (
        os.environ.get('MYCELIA_ROLE') == 'miner' or
        'mycelia/miner/train.py' in call_stack
    )
    
    if is_miner_context and not allow_miner_execution:
        raise PermissionError(
            "ESFT cannot be run by miners!\n"
            "Miners MUST use SN owner's published assignments."
        )
```

**Protection:**
- ✅ Blocks ESFT execution at runtime
- ✅ Checks environment variable
- ✅ Inspects call stack for miner context
- ✅ Raises loud error with explanation

### Layer 2: Environment Marker

**File:** `mycelia/miner/train.py`

```python
# Set flag to prevent ESFT execution
os.environ['MYCELIA_ROLE'] = 'miner'
```

**Protection:**
- ✅ Set at miner startup (cannot be easily bypassed)
- ✅ Checked by ESFT function
- ✅ Persistent throughout miner process lifetime

### Layer 3: Validator Detection

**File:** `mycelia/validator/evaluator.py`

```python
def verify_miner_expert_assignment(...):
    # Check if miner trained MORE experts than authorized
    if total_trained > total_authorized * 1.1:  # 10% tolerance
        return False, "excessive_experts_trained"
```

**Protection:**
- ✅ Detects if miner trained too many experts (ESFT bypass)
- ✅ Compares trained vs authorized expert count
- ✅ Rejects submissions with `score=0`
- ✅ Logs violation for audit

## Attack Scenarios & Defenses

### Scenario 1: Miner Directly Calls ESFT

```python
# Attacker tries:
from mycelia.shared.esft_selector import run_esft_expert_selection
selected = run_esft_expert_selection(model, dataloader, config)
```

**Defense:** Immediate `PermissionError`

```
SECURITY ERROR: ESFT expert selection cannot be run by miners!
Miners MUST use expert assignments provided by the subnet owner.
Running ESFT locally is a protocol violation.
```

### Scenario 2: Miner Modifies Environment Variable

```python
# Attacker tries:
os.environ['MYCELIA_ROLE'] = 'validator'  # Fake validator context
run_esft_expert_selection(...)
```

**Defense:** Call stack inspection still detects miner context

```python
# We also check call stack
if 'mycelia/miner/train.py' in call_stack:
    is_miner_context = True  # Caught!
```

### Scenario 3: Miner Trains Extra Experts

```python
# Attacker runs ESFT in separate script:
python run_esft_locally.py  # Not in miner context
# Then loads those experts into checkpoint
```

**Defense:** Validator detects excess experts

```python
# Validator checks:
total_authorized = 32  # From SN owner
total_trained = 64     # From checkpoint
if 64 > 32 * 1.1:
    reject_submission("excessive_experts_trained")
```

### Scenario 4: Miner Forks Code to Remove Check

```python
# Attacker removes security check from their fork
# def run_esft_expert_selection(...):
#     # is_miner_context check removed
```

**Defense:** Validator still rejects based on expert count

Even if miner bypasses runtime check, validator verification catches the violation:
- Authorized experts: `[0, 1, 2, 3]` (4 experts per layer)
- Trained experts: `[0, 1, 4, 5, 8, 9, ...]` (ESFT selected 12 experts)
- **Result:** Rejected with `score=0`

## Workflow: SN Owner vs Miner

### SN Owner (Authorized)

```python
# 1. Run ESFT on representative data
config = ESFTConfig(
    method="token",
    threshold=0.2,
    save_selection=True,
    selection_path=Path("expert_groups/exp_math/expert_assignment.json"),
)
selected = run_esft_expert_selection(model, dataloader, config)

# 2. Publish to assignment service
assignment_manager.assign_expert_to_miner(
    miner_hotkey="5Abc...",
    expert_group_id=0,
    layer_assignments=selected,
)

# 3. Miners fetch via API (automatically)
```

### Miner (Restricted)

```python
# 1. Fetch assignment from SN owner (done by ExpertManager)
expert_manager = ExpertManager(config)  # Fetches from API
# → GET http://sn_owner:7000/get-expert-assignment

# 2. Train only assigned experts (enforced by model loading)
model = load_model(config, expert_manager)  # Only loads assigned experts

# 3. Submit checkpoint
# → Validator verifies trained experts match assignment
```

## Testing

### Test 1: Miner Attempts ESFT

```python
# test_esft_security.py
import os
os.environ['MYCELIA_ROLE'] = 'miner'

from mycelia.shared.esft_selector import run_esft_expert_selection

try:
    run_esft_expert_selection(model, dataloader, config)
    assert False, "Should have raised PermissionError"
except PermissionError as e:
    assert "ESFT cannot be run by miners" in str(e)
    print("✓ Security check working")
```

### Test 2: Validator Detects Excess Experts

```python
# test_validator_detection.py

# Create checkpoint with unauthorized experts
checkpoint = {
    "model_state_dict": {
        "model.layers.0.mlp.experts.0.gate_proj.weight": ...,
        "model.layers.0.mlp.experts.1.gate_proj.weight": ...,
        "model.layers.0.mlp.experts.99.gate_proj.weight": ...,  # UNAUTHORIZED
    }
}

# Validator rejects
is_valid, reason = verify_miner_expert_assignment(
    miner_hotkey="5Abc...",
    expert_group_id=0,
    checkpoint_path=checkpoint_path,
    config=validator_config,
)

assert not is_valid
assert "unauthorized_experts" in reason
print("✓ Validator detection working")
```

### Test 3: SN Owner Can Run ESFT

```python
# test_sn_owner_esft.py

# No MYCELIA_ROLE env var (or set to 'owner')
os.environ.pop('MYCELIA_ROLE', None)

# Should succeed
selected = run_esft_expert_selection(model, dataloader, config)
print("✓ SN owner can run ESFT")
```

## Monitoring & Alerts

### SN Owner Dashboard

```python
# Monitor for ESFT bypass attempts
def check_for_violations():
    violations = []
    
    for miner_submission in recent_submissions:
        is_valid, reason = verify_expert_assignment(miner_submission)
        
        if not is_valid and "excessive_experts" in reason:
            violations.append({
                "miner": miner_submission.hotkey,
                "reason": reason,
                "timestamp": time.time(),
            })
    
    if violations:
        send_alert(f"{len(violations)} ESFT bypass attempts detected")
```

### Miner Logs

Legitimate miners will see:
```
✓ Fetched expert assignment from SN owner
✓ Expert group 0: 4 experts per layer across 48 layers
✓ Training started with authorized experts only
```

Malicious miners will see:
```
✗ SECURITY ERROR: ESFT expert selection cannot be run by miners!
```

## Summary

| Attack Vector | Defense Mechanism | Result |
|---------------|-------------------|--------|
| Direct ESFT call | Runtime `PermissionError` | ✅ Blocked |
| Env var manipulation | Call stack inspection | ✅ Caught |
| Training extra experts | Validator count check | ✅ Rejected |
| Code fork/bypass | Validator verification | ✅ Score=0 |

**Key Insight:** Even if miners bypass runtime checks, **validators always verify** the final checkpoint against the authoritative SN owner assignment. The three-layer defense ensures **no single bypass point**.

## References

- ESFT Implementation: `mycelia/shared/esft_selector.py`
- Miner Setup: `mycelia/miner/train.py`
- Validator Verification: `mycelia/validator/evaluator.py`
- Full Documentation: `docs/CENTRALIZED_EXPERT_ASSIGNMENT.md`

