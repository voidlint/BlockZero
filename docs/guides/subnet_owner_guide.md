# Subnet Owner Guide

## Overview

The subnet owner coordinates the network, manages expert assignments, and synchronizes training phases.

## Responsibilities

1. **Expert Assignment** - Assign experts to miners
2. **Phase Coordination** - Manage 600-block training cycles
3. **ESFT Execution** - Run expert selection for each task
4. **Network Monitoring** - Track miner/validator health

## Setup

### 1. Start Phase Service

```bash
python mycelia/sn_owner/phase_service.py --path config.yaml
```

**Endpoints:**
- `GET /get_phase` - Current phase info
- `GET /get-expert-assignment` - Miner assignments

### 2. Initialize Expert Assignments

```python
from mycelia.sn_owner.expert_assignment_manager import ExpertAssignmentManager
from mycelia.shared.config import OwnerConfig
from pathlib import Path

config = OwnerConfig()
manager = ExpertAssignmentManager(config)

# Get registered miners from metagraph
import bittensor
subtensor = bittensor.Subtensor(network="test")
metagraph = subtensor.metagraph(netuid=1)
miner_hotkeys = [n.hotkey for n in metagraph.neurons if n.stake > 0]

# Auto-assign experts
manager.auto_assign_miners(
    miner_hotkeys=miner_hotkeys,
    expert_groups_config={
        0: {0: [(0, 0), (1, 1)], 1: [(0, 0), (1, 1)]},  # Math
        1: {0: [(2, 2), (3, 3)], 1: [(2, 2), (3, 3)]},  # Agentic
        2: {0: [(0, 0), (1, 1)], 1: [(0, 0), (1, 1)]},  # Planning
        3: {0: [(2, 2), (3, 3)], 1: [(2, 2), (3, 3)]},  # Vision
    },
    strategy="round_robin",  # or "balanced"
)
```

### 3. Run ESFT for Each Expert Group

```python
from mycelia.shared.esft_selector import run_esft_expert_selection, ESFTConfig
from mycelia.shared.dataloader import get_dataloader

# Math group
config = ESFTConfig(
    method="token",
    threshold=0.2,
    num_samples=32,
    save_selection=True,
    selection_path=Path("expert_groups/exp_math/expert_assignment.json"),
)

selected = run_esft_expert_selection(
    model=model,
    dataloader=math_sample_dataloader,
    config=config,
)

# Repeat for other groups (agentic, planning, vision)
```

## Expert Assignment Strategies

### Round Robin (Default)
Distributes experts evenly in rotation among miners

### Balanced
Attempts equal total expert count per miner

### Performance-Based
Assign more experts to better-performing miners (custom implementation)

## Managing Assignments

### View Current Assignments

```python
assignments = manager.get_all_miner_hotkeys()
for miner in assignments:
    for group_id in manager.get_all_expert_groups():
        assignment = manager.get_miner_assignment(miner, group_id)
        if assignment[group_id]:
            total = sum(len(experts) for experts in assignment[group_id].values())
            print(f"{miner[:16]}... Group {group_id}: {total} experts")
```

### Remove Miner

Edit `sn_owner_expert_assignments.json` or use the manager API to remove assignments.

## Phase Management

### Current Phase

```bash
curl http://localhost:7000/get_phase
```

```json
{
  "phase_name": "train",
  "phase_index": 1,
  "block_start": 100,
  "block_end": 200,
  "current_block": 150,
  "blocks_until_next_phase": 50
}
```

### Phase Timing

```
Block Range     Phase      Duration    Description
0-100          Distribute  100 blocks  Miners download model
100-200        Train       100 blocks  Miners train experts
200-300        Commit      100 blocks  Hash commitments
300-400        Submit      100 blocks  Submit checkpoints
400-500        Validate    100 blocks  Validators evaluate
500-600        Merge       100 blocks  Aggregate updates
```

## Monitoring

### Dashboard

```bash
# Check service health
curl http://localhost:7000/
```

### Key Metrics
- Active miners count
- Validators online
- Submission rate
- Average evaluation time
- Network health score

## Troubleshooting

### Issue: Miners Not Fetching Assignments

**Check service status:**
```bash
curl http://localhost:7000/
```

**Check assignment exists:**
```bash
curl "http://localhost:7000/get-expert-assignment?miner_hotkey=5Abc...&expert_group_id=0&signature=...&timestamp=..."
```

### Issue: Phase Desynchronization

**Restart phase service:**
```bash
# Service recalculates phases from current block
python mycelia/sn_owner/phase_service.py --path config.yaml
```

## Security

### ESFT Execution Control
- Only SN owner can run ESFT
- Miners attempting ESFT are blocked
- Validators verify miner assignments

### Assignment Integrity
- Assignments stored in `sn_owner_expert_assignments.json`
- API-based distribution (not local files)
- Signature verification on requests

## References

- [Centralized Assignment](../CENTRALIZED_EXPERT_ASSIGNMENT.md)
- [ESFT Security](../ESFT_SECURITY.md)
- [ESFT API](../api/esft_api.md)
- [Expert Assignment Manager](../../mycelia/sn_owner/expert_assignment_manager.py)
