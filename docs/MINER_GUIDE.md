# Miner Guide - Mycelia Subnet

Complete guide for running a miner on the Mycelia subnet.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Your Miner](#running-your-miner)
- [Understanding Miner Operation](#understanding-miner-operation)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

As a miner, you train specialized expert modules within a Mixture of Experts model. You'll:

1. **Download** consensus models from validators
2. **Train** your assigned expert modules on domain-specific data
3. **Submit** checkpoints to validators for evaluation
4. **Earn** rewards based on training quality

---

## Prerequisites

### Hardware Requirements

**Minimum**:
- 16GB RAM
- 100GB disk space
- CPU with 4+ cores

**Recommended**:
- 32GB RAM
- 500GB NVMe SSD
- NVIDIA GPU with 16GB+ VRAM (RTX 3090, A5000, or better)
- 8+ CPU cores

### Software Requirements

- Python 3.10 or higher
- CUDA 11.8+ (if using GPU)
- Bittensor wallet

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/subnet-MoE.git
cd subnet-MoE
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install PyTorch with CUDA support (if using GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Setup Bittensor Wallet

```bash
# Create wallet if you don't have one
btcli wallet new_coldkey --wallet.name miner_wallet
btcli wallet new_hotkey --wallet.name miner_wallet --wallet.hotkey miner_hotkey

# Register on subnet (requires TAO)
btcli subnet register --netuid 1 --wallet.name miner_wallet --wallet.hotkey miner_hotkey
```

---

## Configuration

### Create Miner Config

Create `config/miner_config.yaml`:

```yaml
# Role Configuration
role: miner

# Task Configuration - CHOOSE YOUR EXPERT GROUP
task:
  expert_group_id: 0  # 0=Math, 1=Agentic, 2=Planning, 3=Vision, 4=Robotics

# Chain Configuration
chain:
  network: finney  # or 'test' for testnet, 'local' for local dev
  netuid: 1        # Your subnet UID

# Wallet Configuration
wallet_name: miner_wallet
wallet_hotkey_name: miner_hotkey

# Checkpoint Configuration
ckpt:
  checkpoint_path: ./checkpoints          # Where to save your trained checkpoints
  validator_checkpoint_path: ./validator_checkpoints  # Where to download validator models
  resume_from_ckpt: true                 # Resume from last checkpoint if available
  checkpoint_topk: 5                     # Keep last 5 checkpoints

# Training Configuration
training:
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 0.0001
  max_grad_norm: 1.0

# Model Configuration
model:
  device: auto  # auto-detect best device (cuda/mps/cpu)
  use_quantization: false  # Set true for 4-bit quantization (saves memory)
```

### Expert Group Selection

Choose based on your hardware and interests:

| Group ID | Domain | Data Focus | GPU Requirement |
|----------|--------|------------|-----------------|
| 0 | Math | Mathematical problems, equations | 16GB+ |
| 1 | Agentic | Multi-step reasoning, planning | 16GB+ |
| 2 | Planning | Strategic decisions, optimization | 16GB+ |
| 3 | Vision | Images, visual understanding | 24GB+ |
| 4 | Robotics | Control, perception, actions | 16GB+ |

---

## Running Your Miner

### Start Miner

```bash
# With custom config
python -m mycelia.miner.model_io --path config/miner_config.yaml

# With default config
python -m mycelia.miner.model_io
```

### Run as Background Service

**Using systemd** (Linux):

Create `/etc/systemd/system/mycelia-miner.service`:

```ini
[Unit]
Description=Mycelia Miner
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/subnet-MoE
Environment="PATH=/path/to/subnet-MoE/venv/bin"
ExecStart=/path/to/subnet-MoE/venv/bin/python -m mycelia.miner.model_io --path config/miner_config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mycelia-miner
sudo systemctl start mycelia-miner
sudo systemctl status mycelia-miner
```

**Using screen** (Simple):

```bash
screen -S miner
python -m mycelia.miner.model_io --path config/miner_config.yaml
# Press Ctrl+A, then D to detach

# Reattach later
screen -r miner
```

---

## Understanding Miner Operation

### Phase Cycle

Your miner operates in phases synchronized with the blockchain:

```
1. DISTRIBUTE PHASE (100 blocks)
   → Checks validator consensus
   → Downloads new model if ≥60% validators agree
   → If no consensus: stays on current model (fork-safe)

2. TRAIN PHASE (200 blocks)
   → Trains assigned expert modules
   → Saves checkpoints periodically
   → Monitors for NaN/Inf losses

3. COMMIT PHASE (50 blocks)
   → Computes hash of trained model
   → Commits hash to blockchain

4. SUBMISSION PHASE (100 blocks)
   → Finds assigned validator
   → Uploads checkpoint to validator
   → Retries on failure (max 3 attempts)

5. VALIDATE PHASE (100 blocks)
   → Validator evaluates your submission
   → Wait for next cycle

6. MERGE PHASE (50 blocks)
   → Validators aggregate updates
   → New model prepared for distribution
```

### Logs to Monitor

**Download Phase**:
```
[DISTRIBUTE] No consensus or no new model - staying on current model (fork-safe)
[DISTRIBUTE] ✅ Downloaded model with consensus - hash: abc123..., validators agreeing: 5
```

**Training Phase**:
```
[TRAIN] Training step 100/1000, loss: 2.45
[TRAIN] Checkpoint saved: checkpoints/step_500
```

**Submission Phase**:
```
[SUBMISSION] Submitting model to validator xyz...
[SUBMISSION] ✅ Model submitted successfully
```

### What Gets Trained

You train **specific expert modules** assigned to your expert group. For example:

- **Math Group**: Experts specialized in arithmetic, algebra, calculus, etc.
- **Vision Group**: Experts for object detection, segmentation, captioning, etc.

The validator assigns you specific expert IDs (e.g., experts 0, 5, 12) which are deterministic and rotate each cycle.

---

## Troubleshooting

### Issue: "No consensus on model - staying on current model"

**Cause**: Validators haven't agreed on a model hash yet

**Solution**:
- This is normal, especially during network startup
- Your miner will keep training with current model
- Once validators reach consensus, you'll download the new model

**Action**: No action needed - this is fork-safe behavior

---

### Issue: "Checkpoint incompatible: X missing keys"

**Cause**: Downloaded model doesn't match expected architecture

**Solutions**:
1. Check you're running the correct model version
2. Verify your config matches the network
3. Clear old checkpoints: `rm -rf checkpoints/* validator_checkpoints/*`
4. Restart miner to download fresh model

---

### Issue: "Out of memory (OOM)"

**Cause**: Model too large for available GPU memory

**Solutions**:

**Option 1**: Enable quantization
```yaml
model:
  use_quantization: true  # 4-bit quantization
```

**Option 2**: Reduce batch size
```yaml
training:
  batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

**Option 3**: Use gradient checkpointing (already enabled by default)

**Option 4**: Switch to CPU training (slower)
```yaml
model:
  device: cpu
```

---

### Issue: "No validator destination found"

**Cause**: Miner couldn't find assigned validator

**Solutions**:
1. Check network connectivity
2. Verify you're registered on the subnet: `btcli subnet list --netuid 1`
3. Check validators are online in the subnet
4. Wait for next cycle - assignment is deterministic and changes each cycle

---

### Issue: "Signature verification failed"

**Cause**: Downloaded model has invalid signature

**Solution**:
- This is security working correctly
- Miner will reject the model and stay on current
- If persistent, check your system clock is synchronized: `timedatectl`

---

## Best Practices

### 1. Monitor Your Miner

Use monitoring tools to track:
- GPU utilization: `nvidia-smi`
- Disk space: `df -h`
- Logs: `tail -f logs/miner.log`

### 2. Keep Checkpoints Clean

```bash
# Regularly clean old checkpoints (config handles this automatically)
# But manually check if disk space is low
du -sh checkpoints/
```

### 3. Update Regularly

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### 4. Backup Your Wallet

```bash
# Backup wallet keys
cp ~/.bittensor/wallets/miner_wallet ~/wallet_backup/
```

### 5. Use Stable Internet

- Training is long-running (hours)
- Unstable connection can cause failed submissions
- Consider using a VPS with stable connectivity

### 6. Monitor Rewards

```bash
# Check your miner's performance
btcli wallet overview --wallet.name miner_wallet --netuid 1
```

---

## Performance Optimization

### GPU Optimization

```yaml
model:
  device: cuda
  use_quantization: false  # Full precision for best quality

training:
  batch_size: 8  # Max your GPU can handle
  gradient_accumulation_steps: 2
```

### Memory Optimization

```yaml
model:
  use_quantization: true  # 4-bit quantization

training:
  batch_size: 2
  gradient_accumulation_steps: 8  # Maintain effective batch size of 16
```

### CPU Training (Not Recommended)

```yaml
model:
  device: cpu

training:
  batch_size: 1
  gradient_accumulation_steps: 16
```

---

## FAQ

**Q: How often should I expect rewards?**
A: Rewards are distributed after validation phases when your submissions are accepted and score well.

**Q: Can I run multiple miners?**
A: Yes, but you need separate wallets and hardware for each.

**Q: What happens if I go offline?**
A: You'll miss that cycle's submission. Resume when you're back online.

**Q: How do I switch expert groups?**
A: Change `expert_group_id` in config and restart. You'll start training different experts next cycle.

**Q: Is my data used for training?**
A: No, you train on standardized datasets provided by the subnet. Your local data is not used.

---

## Support

- Check logs first: `tail -n 100 logs/miner.log`
- Search GitHub issues
- Ask in Discord support channel
- File a bug report with logs attached

---

## Next Steps

1. ✅ Configure your miner
2. ✅ Start mining
3. ✅ Monitor logs
4. ✅ Track rewards
5. ✅ Optimize performance

Happy mining! 🚀
