# Centralized Expert Assignment System

## Overview

This system enforces **SN owner control** over expert assignments, preventing miners from self-selecting which experts to train. This is critical for:

1. **Fair distribution** - Ensures experts are distributed evenly
2. **Security** - Prevents gaming by choosing "easy" experts
3. **Quality control** - SN owner can assign based on miner performance
4. **Accountability** - Clear audit trail of who trains what

## Architecture

```
┌──────────────┐
│  SN Owner    │
│   Service    │ ← Authoritative source of expert assignments
└──────┬───────┘
       │
       │ GET /get-expert-assignment?miner=...&group=...
       │
    ┌──┴──────────────┐
    │                 │
┌───▼────┐      ┌────▼────┐
│ Miner  │      │Validator│
│        │      │         │
│ Fetches│      │Verifies │
│ on     │      │before   │
│ startup│      │scoring  │
└────────┘      └─────────┘
```

## Components

### 1. SN Owner Service (`mycelia/sn_owner/`)

**`expert_assignment_manager.py`** - Manages expert-to-miner assignments

```python
manager = ExpertAssignmentManager(config)

# Auto-assign experts to registered miners
manager.auto_assign_miners(
    miner_hotkeys=["5Abc...", "5Def..."],
    base_path=Path("expert_groups/"),
    strategy="round_robin",  # or "balanced"
)

# Get assignment for specific miner
assignment = manager.get_assignment("5Abc...", expert_group_id=0)
```

**`phase_service.py`** - Serves assignments via API

```bash
# Start the service
python mycelia/sn_owner/phase_service.py --path config.yaml

# API endpoint
GET http://localhost:7000/get-expert-assignment?miner_hotkey=5Abc...&expert_group_id=0
```

### 2. Miner (`mycelia/miner/`)

**`ExpertManager.load_expert_group_assignment()`** - Fetches from SN owner

```python
# In miner startup:
expert_manager = ExpertManager(config)  # Automatically fetches from SN owner
# ✓ Assignment fetched from http://149.137.225.62:7000
# ✓ Miner 5Abc... assigned 8 experts in 48 layers
```

**No local file modification possible** - Assignments are fetched at runtime

### 3. Validator (`mycelia/validator/`)

**`verify_miner_expert_assignment()`** - Validates submissions

```python
# Before evaluating miner model:
is_valid, reason = verify_miner_expert_assignment(
    miner_hotkey="5Abc...",
    expert_group_id=0,
    checkpoint_path=Path("miner_submission.pt"),
    config=validator_config,
)

if not is_valid:
    # Reject submission with score=0
    logger.error(f"Rejected: {reason}")
```

## Setup Instructions

### For SN Owner

1. **Configure storage path** in `config.yaml`:
```yaml
owner:
  assignment_storage_path: "data/expert_assignments"
```

2. **Generate assignments**:
```python
from mycelia.sn_owner.expert_assignment_manager import ExpertAssignmentManager
from mycelia.shared.config import OwnerConfig

config = OwnerConfig.from_path("config.yaml")
manager = ExpertAssignmentManager(config)

# Get registered miners from metagraph
import bittensor
subtensor = bittensor.Subtensor(network="test")
metagraph = subtensor.metagraph(netuid=1)
miner_hotkeys = [n.hotkey for n in metagraph.neurons if n.stake > 0]

# Auto-assign
manager.auto_assign_miners(
    miner_hotkeys=miner_hotkeys,
    base_path=Path("expert_groups/"),
    strategy="round_robin",
)
```

3. **Start API service**:
```bash
python mycelia/sn_owner/phase_service.py --path config.yaml
```

### For Miners

1. **Configure owner URL** in `checkpoints/miner/testnet/default/1/config.yaml`:
```yaml
owner_url: "http://149.137.225.62:7000"
role: "miner"
```

2. **Run miner** - Assignment is fetched automatically:
```bash
python -m mycelia.miner.run --path checkpoints/miner/testnet/default/1/config.yaml
```

3. **Error handling**:
```
✗ Failed to fetch expert assignment from SN owner
  Error: 403 Forbidden - Miner not authorized for expert group 0
  
  → Contact subnet owner for assignment
```

### For Validators

1. **Configure owner URL** in validator config:
```yaml
owner_url: "http://149.137.225.62:7000"
```

2. **Run validator** - Verification happens automatically:
```bash
python -m mycelia.validator.run --path config.yaml
```

3. **Rejection logs**:
```
✗ Miner 5Abc... trained unauthorized experts
  Layer 5: unauthorized=[4, 5] | authorized=[0, 1, 2, 3]
  → Score: 0.0 | Status: rejected
```

## Assignment Strategies

### Round Robin (Default)
Distributes experts evenly in rotation:
- Miner 1: Experts 0, 3, 6, 9, ...
- Miner 2: Experts 1, 4, 7, 10, ...
- Miner 3: Experts 2, 5, 8, 11, ...

### Balanced
Attempts to give each miner the same total expert count across all groups.

### Custom
Implement your own in `ExpertAssignmentManager`:
```python
def assign_by_performance(
    self,
    miners: List[str],
    performance_scores: Dict[str, float],
):
    # Assign "harder" experts to better miners
    ...
```

## Security Features

### 1. API-Based Assignment
- Miners **cannot** modify local `expert_assignment.json`
- Assignments fetched at runtime from authoritative source
- Changes require SN owner update

### 2. ESFT Execution Restriction
- **Miners are BLOCKED from running ESFT** (`run_esft_expert_selection()`)
- ESFT can only be run by SN owner in sn_owner context
- Attempting to run ESFT as a miner raises `PermissionError`
- Environment variable `MYCELIA_ROLE=miner` is set automatically
- Call stack inspection detects miner context

**Penalty for violation:**
```
✗ SECURITY ERROR: ESFT expert selection cannot be run by miners!
  → Submission rejection by validators
  → Zero rewards
  → Potential blacklisting
```

### 3. Validator Verification
- Extracts trained experts from checkpoint
- Compares against authorized assignment
- Checks total expert count (detects ESFT bypass)
- Rejects mismatches with score=0

**Verification checks:**
1. All trained experts are in authorized list
2. No unauthorized experts present
3. Total expert count within tolerance (±10%)

### 4. Signature Verification (Optional)
```python
from mycelia.shared.signature_utils import verify_assignment_signature

# Verify SN owner signed the assignment
is_authentic = verify_assignment_signature(
    assignment=assignment_data,
    sn_owner_hotkey=config.sn_owner_hotkey,
)
```

## Troubleshooting

### Miner: "Failed to fetch assignment"
**Cause**: SN owner service not running or unreachable

**Fix**:
```bash
# Check service is running
curl http://149.137.225.62:7000/

# Check assignment exists
curl "http://149.137.225.62:7000/get-expert-assignment?miner_hotkey=5Abc...&expert_group_id=0"
```

### Miner: "403 Not authorized"
**Cause**: Miner not assigned by SN owner

**Fix**: Contact SN owner to add your hotkey to assignments

### Validator: "Verification error"
**Cause**: SN owner service unreachable during validation

**Behavior**: Fail open (allows evaluation) - Change to fail closed in production

**Fix**: Ensure SN owner service has high uptime

## API Reference

### `GET /get-expert-assignment`

**Parameters:**
- `miner_hotkey` (string): Miner's SS58 address
- `expert_group_id` (int): Expert group (0=math, 1=agentic, etc.)

**Response:**
```json
{
  "miner_hotkey": "5AbcDef...",
  "expert_group_id": 0,
  "layer_assignments": {
    "0": [[0, 0], [1, 4]],
    "1": [[0, 1], [1, 5]],
    ...
  },
  "timestamp": 1704067200.0
}
```

**Errors:**
- `403`: Miner not authorized
- `404`: No assignment found
- `500`: Internal error

## Migration from Local Files

If you have existing local assignments:

1. **Validators/SN Owner**: No change needed (still loads from files)
2. **Miners**: Remove local `expert_assignment.json` files (they're ignored now)
3. **Generate assignments**: Run `auto_assign_miners()` once to populate database

## Future Enhancements

- [ ] Dynamic reassignment based on performance
- [ ] Assignment versioning/history
- [ ] Automatic rebalancing when miners join/leave
- [ ] Full cryptographic signature verification
- [ ] Assignment expiration/renewal
- [ ] Web UI for assignment management

## References

- `mycelia/sn_owner/expert_assignment_manager.py` - Assignment logic
- `mycelia/sn_owner/phase_service.py` - API endpoint
- `mycelia/shared/expert_manager.py` - Miner assignment fetching
- `mycelia/validator/evaluator.py` - Validator verification
- `mycelia/shared/signature_utils.py` - Signature utilities

