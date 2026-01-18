# Miner Setup Guide

Complete guide to setting up and running a miner node in the MoE subnet.

## Prerequisites

### Hardware Requirements

**Minimum (with quantization):**
- GPU: 1x A6000 (48GB VRAM)
- RAM: 64GB
- Storage: 200GB SSD
- Network: 100 Mbps

**Recommended:**
- GPU: 1x A100 80GB or 2x A6000 48GB
- RAM: 128GB
- Storage: 500GB NVMe SSD
- Network: 1 Gbps

**Not Supported:**
- Mac M-series (max 24GB unified memory insufficient)
- Consumer GPUs (<24GB VRAM)
- CPU-only setups

### Software Requirements

```bash
# System
Ubuntu 22.04 LTS (recommended)
Python 3.10+
CUDA 12.1+

# Dependencies
torch>=2.1.0
transformers>=4.36.0
bitsandbytes>=0.41.0
bittensor>=6.0.0
```

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/CognitoBlocks/BlockZero.git
cd BlockZero
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 4: Install Bittensor

```bash
# Install bittensor
pip install bittensor

# Create wallet
btcli wallet create --wallet.name miner_wallet --wallet.hotkey miner_hotkey

# Register on testnet
btcli subnet register \
  --netuid 1 \
  --subtensor.network test \
  --wallet.name miner_wallet \
  --wallet.hotkey miner_hotkey
```

## Configuration

### Step 1: Create Expert Group Config

Each expert group needs a `config.yaml` file:

```bash
cd expert_groups

# Choose your expert group:
# - exp_math (Group 0): Mathematical reasoning
# - exp_agentic (Group 1): Tool use, function calling
# - exp_planning (Group 2): Multi-step planning
# - exp_vision (Group 3): Visual understanding

# Copy template
cp config.template.yaml exp_math/config.yaml
```

### Step 2: Edit Expert Group Config

```yaml
# expert_groups/exp_math/config.yaml

# Expert Group Identity
expert_group_id: 0
expert_group_name: "exp_math"

# Data Configuration
data:
  dataset_name: "merged_math"
  dataset_class: "expert_groups.exp_math.dataset:MergedMathDataset"
  batch_size: 512
  sequence_length: 2048
  per_device_train_batch_size: 2
  world_size: 1  # Single GPU
  rank: 0

# Model Configuration (inherited from main config)
```

### Step 3: Create Miner Config

Generate miner configuration:

```bash
python mycelia/shared/config.py \
  --get_template miner \
  --out_path checkpoints/miner/testnet/my_miner/1/config.yaml
```

### Step 4: Edit Miner Config

```yaml
# checkpoints/miner/testnet/my_miner/1/config.yaml

# Role
role: "miner"

# Subnet Owner URL (IMPORTANT)
owner_url: "http://149.137.225.62:7000"

# Task Configuration
task:
  base_path: "expert_groups/"
  expert_group_id: 0  # Math group

# Wallet Configuration
wallet:
  name: "miner_wallet"
  hotkey: "miner_hotkey"

# Chain Configuration
chain:
  netuid: 1
  network: "test"  # or "main" for mainnet

# Model Configuration
model:
  model_path: "Qwen/Qwen3-VL-30B-A3B-Instruct"
  use_quantization: true  # REQUIRED for 48GB GPUs
  torch_compile: false
  precision: "fp16-mixed"

# Training Configuration
opt:
  lr: 1e-5
  outer_lr: 1e-4
  outer_momentum: 0.9

sched:
  warmup_steps: 100
  total_steps: 10000

# Checkpoint Configuration
ckpt:
  checkpoint_path: "checkpoints/miner/testnet/my_miner/1"
  resume_from_ckpt: true
  skip_network_sync: false  # IMPORTANT: Fetch from validators
  keep_n_checkpoints: 5
```

## Running the Miner

### Step 1: Verify Configuration

```bash
# Test configuration loading
python -c "
from mycelia.shared.config import MinerConfig
config = MinerConfig.from_path('checkpoints/miner/testnet/my_miner/1/config.yaml')
print('✓ Config loaded successfully')
print(f'Expert Group: {config.task.expert_group_id}')
print(f'Subnet Owner: {config.owner_url}')
"
```

### Step 2: Start Miner

```bash
# Single GPU
python -m mycelia.miner.run \
  --path checkpoints/miner/testnet/my_miner/1/config.yaml

# Multi-GPU (optional)
python -m mycelia.miner.run \
  --path checkpoints/miner/testnet/my_miner/1/config.yaml \
  --local_par.world_size 2
```

### Expected Output

```
✓ Loaded config from checkpoints/miner/testnet/my_miner/1/config.yaml
✓ Wallet: miner_wallet (hotkey: 5Abc...)
✓ Registered on subnet 1 (testnet)
✓ Fetched expert assignment from SN owner
  → Expert group 0 (Math)
  → 96 experts across 48 layers
✓ Model loaded: Qwen3-VL-30B-A3B-Instruct (4-bit quantized)
✓ Trainable parameters: 8.2B / 30B (27.3%)
✓ Training started

Phase: DISTRIBUTE (blocks 0-100)
  ↓ Downloading latest model from validator...
  ↓ Model version: 42
  ✓ Model downloaded and loaded

Phase: TRAIN (blocks 100-200)
  Step 1/1000 | Loss: 2.453 | LR: 1e-6 | Time: 3.2s
  Step 2/1000 | Loss: 2.389 | LR: 2e-6 | Time: 3.1s
  ...
```

## Monitoring

### Logs

```bash
# View logs
tail -f logs/miner_*.log

# Key metrics to monitor:
# - Loss (should decrease)
# - Training time per batch (3-5s typical)
# - Memory usage (should be <80% of GPU)
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard

# Open browser: http://localhost:6006

# Metrics:
# - train/loss
# - train/learning_rate
# - train/grad_norm
```

### GPU Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Expected:
# GPU 0: 40-45 GB / 48 GB
# Utilization: 80-100%
# Temperature: 60-80°C
```

## Troubleshooting

### Issue 1: Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. Enable quantization:
```yaml
model:
  use_quantization: true
```

2. Reduce batch size:
```yaml
data:
  per_device_train_batch_size: 1  # Down from 2
```

3. Enable gradient checkpointing (automatic if enabled)

4. Reduce sequence length:
```yaml
data:
  sequence_length: 1024  # Down from 2048
```

### Issue 2: Failed to Fetch Expert Assignment

**Error:**
```
✗ Failed to fetch expert assignment from SN owner
  Error: Connection refused
```

**Solutions:**

1. Check SN owner service is running:
```bash
curl http://149.137.225.62:7000/
# Should return: {"message": "Phase service is running"}
```

2. Verify your hotkey is assigned:
```bash
curl "http://149.137.225.62:7000/get-expert-assignment?miner_hotkey=5Abc...&expert_group_id=0"
```

3. Contact subnet owner to add your hotkey

### Issue 3: Model Download Timeout

**Error:**
```
Timeout while downloading model from validator
```

**Solutions:**

1. Check network connectivity:
```bash
ping 149.137.225.62
```

2. Increase timeout:
```yaml
ckpt:
  download_timeout: 600  # 10 minutes
```

3. Use local model cache (if available):
```yaml
model:
  local_cache_path: "/raid/models/qwen3_vl"
```

### Issue 4: Training Not Starting

**Error:**
```
Phase: DISTRIBUTE
Waiting for training phase...
(stuck)
```

**Solution:**

Check phase timing:
```bash
curl http://149.137.225.62:7000/blocks_until_next_phase
```

Wait for TRAIN phase to begin (synchronized across network)

### Issue 5: Submission Rejected

**Error:**
```
✗ Submission rejected by validator
  Reason: unauthorized_experts
```

**Solution:**

This means you trained experts not assigned to you. This is a **security violation**.

1. Verify you're fetching assignment from SN owner (not using local files)
2. Check logs for expert assignment at startup
3. Do NOT run ESFT locally as a miner

## Performance Optimization

### 1. Mixed Precision Training

```yaml
model:
  precision: "fp16-mixed"  # Faster training, less memory
```

### 2. Gradient Accumulation

```yaml
opt:
  gradient_accumulation_steps: 4  # Effective batch size = 2×4 = 8
```

### 3. Compile Model (PyTorch 2.0+)

```yaml
model:
  torch_compile: true  # 20-30% speedup
```

### 4. Optimize Data Loading

```yaml
data:
  num_workers: 4  # Parallel data loading
  pin_memory: true  # Faster GPU transfer
```

### 5. Use NVMe for Checkpoints

```bash
# Move checkpoints to NVMe
mkdir /nvme/checkpoints
ln -s /nvme/checkpoints checkpoints
```

## Rewards & Economics

### Reward Calculation

```python
reward = (
    0.7 * task_performance +      # Primary metric (loss, accuracy)
    0.2 * expert_quality +         # Router activation patterns
    0.1 * participation_bonus      # Uptime, consistency
)
```

### Expected Earnings

| Performance | Daily TAO | Monthly TAO | Requirements |
|-------------|-----------|-------------|--------------|
| Top 10%     | 5-8 TAO   | 150-240 TAO | A100 80GB    |
| Top 25%     | 3-5 TAO   | 90-150 TAO  | A6000 48GB   |
| Average     | 1-3 TAO   | 30-90 TAO   | A6000 48GB   |

### Cost Breakdown

```
Hardware:
- GPU rental: $0.50-2.00/hour
- Electricity: ~$50/month

Network:
- Registration fee: 1 TAO
- Transaction fees: ~0.01 TAO/day

Total: ~$400-1500/month

Break-even: Top 25% performance
```

## Best Practices

### 1. Monitor Phase Timing

```python
import requests

def get_current_phase():
    r = requests.get("http://149.137.225.62:7000/get_phase")
    return r.json()

phase = get_current_phase()
print(f"Current: {phase['phase_name']}")
print(f"Blocks remaining: {phase['blocks_until_next_phase']}")
```

### 2. Backup Checkpoints

```bash
# Daily backup
rsync -av checkpoints/ backup/checkpoints-$(date +%Y%m%d)/
```

### 3. Update Regularly

```bash
# Check for updates
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

### 4. Log Rotation

```bash
# Add to crontab
0 0 * * * find logs/ -name "*.log" -mtime +7 -delete
```

## Advanced Topics

### Multi-Expert Group Mining

Train multiple expert groups simultaneously:

```bash
# Terminal 1: Math expert
python -m mycelia.miner.run --path checkpoints/miner_math/config.yaml

# Terminal 2: Vision expert
CUDA_VISIBLE_DEVICES=1 python -m mycelia.miner.run --path checkpoints/miner_vision/config.yaml
```

### Custom Dataset

Create custom dataset for your expert group:

```python
# expert_groups/exp_custom/dataset.py
from mycelia.shared.dataloader import BaseDataset

class CustomDataset(BaseDataset):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        # Load your data
        self.data = load_custom_data()
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Tokenize
        return self.tokenizer(
            item["text"],
            return_tensors="pt",
            max_length=self.config.sequence_length,
            truncation=True,
        )
```

## Support

- **Documentation:** https://github.com/CognitoBlocks/BlockZero/docs
- **Discord:** https://discord.gg/bittensor
- **Issues:** https://github.com/CognitoBlocks/BlockZero/issues

## Next Steps

- See [Validator Setup](validator_setup.md) to run a validator
- See [MoE Architecture](../architecture/moe_architecture.md) for system details
- See [ESFT Security](../ESFT_SECURITY.md) for security best practices

