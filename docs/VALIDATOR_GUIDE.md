# Validator Guide - Mycelia Subnet

Complete guide for running a validator on the Mycelia subnet.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Your Validator](#running-your-validator)
- [Understanding Validator Operation](#understanding-validator-operation)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

As a validator, you:

1. **Coordinate** the decentralized training process
2. **Evaluate** miner submissions on unpredictable validation sets
3. **Aggregate** updates from top-performing miners
4. **Distribute** new model versions with consensus
5. **Earn** rewards for maintaining network integrity

---

## Prerequisites

### Hardware Requirements

**Minimum**:
- 32GB RAM
- 500GB NVMe SSD
- 8+ CPU cores
- Stable internet (99.9% uptime)

**Recommended**:
- 64GB RAM
- 1TB NVMe SSD
- NVIDIA GPU with 24GB+ VRAM (RTX 4090, A6000)
- 16+ CPU cores
- 1Gbps internet connection

### Software Requirements

- Python 3.10+
- CUDA 11.8+ (if using GPU)
- Bittensor wallet with validator permit
- Public IP or port forwarding configured

### Stake Requirements

- Sufficient TAO staked to obtain validator permit
- Check current requirements: `btcli subnet list --netuid 1`

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/subnet-MoE.git
cd subnet-MoE
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Setup Validator Wallet

```bash
# Create validator wallet
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey validator_hotkey

# Stake TAO to get validator permit
btcli stake add --wallet.name validator --wallet.hotkey validator_hotkey --amount 1000

# Register on subnet
btcli subnet register --netuid 1 --wallet.name validator --wallet.hotkey validator_hotkey
```

### 4. Configure Firewall

```bash
# Allow incoming connections on validator port
sudo ufw allow 8091/tcp  # Default validator port
sudo ufw enable
```

---

## Configuration

### Create Validator Config

Create `config/validator_config.yaml`:

```yaml
# Role Configuration
role: validator

# Task Configuration
task:
  expert_group_id: 0  # Which expert group to validate (0=Math, 1=Agentic, etc.)

# Chain Configuration
chain:
  network: finney
  netuid: 1

# Wallet Configuration
wallet_name: validator
wallet_hotkey_name: validator_hotkey

# Cycle Configuration - MUST MATCH NETWORK CONSENSUS
cycle:
  cycle_length: 600  # Total blocks per cycle
  distribute_period: 100
  train_period: 200
  commit_period: 50
  submission_period: 100
  validate_period: 100
  merge_period: 50

# Server Configuration
server:
  host: 0.0.0.0  # Listen on all interfaces
  port: 8091     # Validator port (must be publicly accessible)

# Checkpoint Configuration
ckpt:
  checkpoint_path: ./validator_checkpoints
  checkpoint_topk: 10  # Keep last 10 checkpoints

# Validation Configuration
validation:
  batch_size: 4
  num_validation_samples: 100  # Samples per evaluation

# Model Configuration
model:
  device: auto
  use_quantization: false
```

### ⚠️ Critical: Consensus Parameters

These parameters **MUST match** across all validators:

```yaml
cycle:
  cycle_length: 600
  distribute_period: 100
  train_period: 200
  commit_period: 50
  submission_period: 100
  validate_period: 100
  merge_period: 50

chain:
  netuid: 1
```

**Verify consensus**:
```bash
python -c "
from mycelia.shared.config import ValidatorConfig
from mycelia.shared.config_validation import log_consensus_critical_config

config = ValidatorConfig.from_path('config/validator_config.yaml')
log_consensus_critical_config(config)
"
```

All validators must have the same config hash.

---

## Running Your Validator

### Start Validator

```bash
# With custom config
python -m mycelia.validator.run --path config/validator_config.yaml

# With default config
python -m mycelia.validator.run
```

### Run as systemd Service

Create `/etc/systemd/system/mycelia-validator.service`:

```ini
[Unit]
Description=Mycelia Validator
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/subnet-MoE
Environment="PATH=/path/to/subnet-MoE/venv/bin"
ExecStart=/path/to/subnet-MoE/venv/bin/python -m mycelia.validator.run --path config/validator_config.yaml
Restart=always
RestartSec=10
StandardOutput=append:/var/log/mycelia-validator.log
StandardError=append:/var/log/mycelia-validator-error.log

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mycelia-validator
sudo systemctl start mycelia-validator
sudo systemctl status mycelia-validator
```

### Monitor Logs

```bash
# Follow logs
tail -f /var/log/mycelia-validator.log

# Check for errors
tail -f /var/log/mycelia-validator-error.log
```

---

## Understanding Validator Operation

### Phase Cycle

Validators operate in a synchronized cycle:

```
CYCLE: 600 blocks (~2 hours)

1. COMMIT PHASE (50 blocks)
   → Commit deterministic seed for miner assignment
   → Seed determines which miners you'll evaluate
   → All validators compute same assignments (decentralized)

2. VALIDATE PHASE (100 blocks)
   → Receive miner submissions
   → Evaluate on unpredictable validation set
   → Score submissions based on loss/accuracy

3. MERGE PHASE (50 blocks)
   → Aggregate top-performing miner updates
   → Update global model with aggregated gradients
   → Save new checkpoint

4. DISTRIBUTE PHASE (100 blocks)
   → Commit new model hash with cryptographic signature
   → Make model available for download
   → Miners check consensus before downloading

5. TRAIN PHASE (200 blocks)
   → Miners train with new model
   → Wait for next cycle
```

### Consensus Mechanism

**Seed Commit (Phase 1)**:
```
Each validator commits:
- commit_phase: 1
- miner_seed: deterministic_seed (from anchor block hash)
- block: current_block

Purpose: Synchronize miner assignments across validators
```

**Model Commit (Phase 2)**:
```
Each validator commits:
- commit_phase: 2
- model_hash: SHA256 of checkpoint
- signed_model_hash: domain-separated signature
- block: current_block

Signature format:
  "mycelia:v1|{netuid}|{expert_group}|{anchor_block_hash}|2|{model_hash}"

Purpose: Achieve consensus on model before distribution
```

**Consensus Requirements**:
- **Quorum**: ≥60% of active validators must commit
- **Majority**: >50% must agree on same model hash
- If not met: miners stay on last finalized model (fork-safe)

### Miner Assignment

Deterministic assignment each cycle:

1. Compute anchor block: `cycle_number * cycle_length`
2. Get anchor block hash from chain
3. Seed RNG: `Random(int(anchor_hash[:16], 16))`
4. Shuffle miner list deterministically
5. Assign miners to validators round-robin

**All validators compute identical assignments** - no coordination needed.

---

## Monitoring

### Check Validator Status

```bash
# Current phase
curl http://localhost:8091/status

# Blocks until next phase
curl http://localhost:8091/phase_info

# Recent evaluations
curl http://localhost:8091/recent_evaluations
```

### Monitor Performance

```bash
# GPU utilization
nvidia-smi

# Network connections (should see miner connections)
netstat -an | grep 8091

# Disk space
df -h

# Check validator weights
btcli subnet list --netuid 1 | grep your_hotkey
```

### Key Metrics

Monitor these in logs:

- **Miner submissions received**: Should receive multiple per cycle
- **Evaluation completion rate**: Should evaluate most submissions
- **Consensus achievement**: Should achieve consensus each cycle
- **Model distribution**: Miners should download your model

---

## Troubleshooting

### Issue: "No miner submissions received"

**Causes**:
1. Port 8091 not publicly accessible
2. Not enough miners online
3. Firewall blocking connections

**Solutions**:
```bash
# Check port is open
sudo netstat -tlnp | grep 8091

# Test external access
curl http://your_public_ip:8091/status

# Check firewall
sudo ufw status
sudo ufw allow 8091/tcp
```

---

### Issue: "Consensus not achieved"

**Cause**: Validators don't agree on model hash

**Solutions**:
1. Check config matches other validators
2. Verify you're using correct cycle parameters
3. Check logs for signature verification errors
4. Ensure your system clock is synchronized:
   ```bash
   timedatectl
   sudo timedatectl set-ntp true
   ```

---

### Issue: "Signature verification failed"

**Cause**: Miner submitted invalid signature or tampered checkpoint

**Solution**:
- This is normal security - invalid submissions are rejected
- Check miner hotkeys in logs
- If persistent from same miner, they may have issues

---

### Issue: "Out of memory during evaluation"

**Solutions**:

1. **Enable quantization**:
   ```yaml
   model:
     use_quantization: true
   ```

2. **Reduce batch size**:
   ```yaml
   validation:
     batch_size: 2
   ```

3. **Reduce validation samples**:
   ```yaml
   validation:
     num_validation_samples: 50
   ```

---

### Issue: "Config hash mismatch"

**Cause**: Your config doesn't match network consensus

**Solution**:
```bash
# Get expected config from another validator or docs
# Update your config to match

# Verify hash
python -c "
from mycelia.shared.config import ValidatorConfig
from mycelia.shared.config_validation import compute_config_hash

config = ValidatorConfig.from_path('config/validator_config.yaml')
print(f'Config hash: {compute_config_hash(config)}')
"

# Compare with other validators
```

---

## Best Practices

### 1. High Availability

- Use stable hosting (AWS, GCP, dedicated server)
- Set up automatic restarts (systemd)
- Monitor uptime (>99.9% recommended)
- Have backup power/internet

### 2. Security

- Keep wallet keys secure and backed up
- Use firewall to limit exposure
- Regularly update dependencies
- Monitor for suspicious submissions

### 3. Performance

- Use SSD for fast checkpoint I/O
- Ensure good network bandwidth for model distribution
- Monitor resource usage
- Optimize evaluation batch size for your GPU

### 4. Monitoring

Set up alerts for:
- Service downtime
- Disk space <20%
- High error rate in logs
- Consensus failures

### 5. Regular Maintenance

```bash
# Weekly: Update code
git pull origin main
pip install -r requirements.txt --upgrade
sudo systemctl restart mycelia-validator

# Monthly: Clean old checkpoints
find validator_checkpoints/ -mtime +30 -delete

# Check validator performance
btcli wallet overview --wallet.name validator --netuid 1
```

---

## Advanced Configuration

### Multi-Expert Group Validation

Run multiple validators for different expert groups:

```bash
# Validator 1: Math group
python -m mycelia.validator.run --path config/validator_math.yaml

# Validator 2: Vision group
python -m mycelia.validator.run --path config/validator_vision.yaml
```

Each needs separate wallet and port.

### Custom Evaluation

Modify validation dataset in `expert_groups/exp_{group}/dataset.py` to add domain-specific evaluation tasks.

---

## FAQ

**Q: How much TAO do I need to stake?**
A: Check current validator requirements: `btcli subnet list --netuid 1`. Typically 1000+ TAO.

**Q: Can I run validator on cloud?**
A: Yes, recommended. AWS/GCP instances with GPU are ideal.

**Q: What happens if I go offline?**
A: You miss consensus commits and miner evaluations. Your validator weight may decrease.

**Q: How are rewards calculated?**
A: Based on validator weight, which depends on consensus participation and evaluation quality.

**Q: Can I validate multiple expert groups?**
A: Yes, run separate validator instances with different configs and wallets.

---

## Support

- Check logs: `tail -n 100 /var/log/mycelia-validator.log`
- GitHub issues
- Discord validator channel
- Community forums

---

## Next Steps

1. ✅ Configure your validator
2. ✅ Start validating
3. ✅ Monitor consensus participation
4. ✅ Optimize performance
5. ✅ Track rewards

Happy validating! 🚀
