# Validator Setup Guide

## Overview

Validators evaluate miner submissions, aggregate expert updates, and maintain the global model.

## Hardware Requirements

**Minimum:**
- GPU: 1x A6000 (48GB) or A100 (80GB)
- RAM: 64GB
- Storage: 500GB SSD
- Network: 1 Gbps (critical for model distribution)

## Quick Start

### 1. Installation

```bash
git clone https://github.com/CognitoBlocks/BlockZero.git
cd BlockZero
pip install -r requirements.txt
```

### 2. Create Wallet

```bash
btcli wallet create --wallet.name validator_wallet --wallet.hotkey validator_hotkey
btcli subnet register --netuid 1 --wallet.name validator_wallet
```

### 3. Configure

```bash
python mycelia/shared/config.py --get_template validator --out_path config.yaml
```

Edit `config.yaml`:
```yaml
role: "validator"
owner_url: "http://149.137.225.62:7000"

wallet:
  name: "validator_wallet"
  hotkey: "validator_hotkey"

model:
  use_quantization: true  # For 48GB GPUs
```

### 4. Run

```bash
python -m mycelia.validator.run --path config.yaml
```

## Validator Responsibilities

### 1. Model Distribution
Serve latest global model to miners during DISTRIBUTE phase

### 2. Submission Evaluation
- Receive miner checkpoints
- **Verify expert assignment** (security check)
- Compute task-specific metrics
- Score submissions

### 3. Expert Aggregation
Merge best expert updates into global model using weighted averaging

### 4. Inter-Validator Consensus
Sync with other validators via DHT (Distributed Hash Table)

## Key Configuration

```yaml
# Checkpoint paths
ckpt:
  checkpoint_path: "checkpoints/validator/testnet/1"
  miner_submission_path: "checkpoints/submissions"
  
# Evaluation
eval:
  max_eval_batches: 50
  concurrent_evaluations: 4
  
# Validator network
hivemind:
  initial_peers: ["/ip4/149.137.225.62/tcp/9000/p2p/..."]
```

## Monitoring

### Logs
```bash
tail -f logs/validator_*.log
```

### Key Metrics
- Miners evaluated per cycle
- Average evaluation time
- Global model version
- Validator consensus status

## Troubleshooting

See [Miner Setup - Troubleshooting](miner_setup.md#troubleshooting) for common issues.

**Validator-specific:**

**Issue:** No miner submissions
- Check phase timing: `curl http://sn_owner:7000/get_phase`
- Verify miners are registered: check metagraph

**Issue:** Consensus failures
- Check DHT connectivity: `ping validator_peer_ip`
- Verify initial peers in config

## References

- [MoE Architecture](../architecture/moe_architecture.md)
- [Metrics API](../api/metrics_api.md)
- [Centralized Assignment](../CENTRALIZED_EXPERT_ASSIGNMENT.md)
