# Mycelia Architecture

Technical architecture documentation for the Mycelia decentralized training subnet.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Bittensor Blockchain                        │
│  • Decentralized State (commits, stakes, permits)               │
│  • Consensus Layer (validator consensus on model hashes)        │
│  • Incentive Mechanism (rewards distribution)                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐       ┌────────▼────────┐
│   Validators   │       │     Miners      │
│                │       │                 │
│ Responsibilities:      │ Responsibilities:│
│ • Coordinate cycles    │ • Train experts │
│ • Evaluate miners      │ • Submit models │
│ • Aggregate updates    │ • Specialize    │
│ • Distribute models    │                 │
│ • Achieve consensus    │                 │
└────────────────┘       └─────────────────┘
```

---

## Component Architecture

### 1. Validators

**Core Components**:
```
mycelia/validator/
├── run.py              # Main validator loop
├── server.py           # HTTP server for miner submissions
├── evaluator.py        # Miner evaluation logic
└── inter_validator_connection.py  # Validator coordination
```

**Validator Flow**:
```
┌────────────────────────────────────────────┐
│  Validator Main Loop (validator/run.py)   │
└──────────────┬─────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │  COMMIT PHASE       │
    │  • Compute seed     │
    │  • Commit to chain  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  VALIDATE PHASE     │
    │  • Receive miners   │
    │  • Evaluate models  │
    │  • Score miners     │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  MERGE PHASE        │
    │  • Aggregate updates│
    │  • Update global    │
    │  • Save checkpoint  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  DISTRIBUTE PHASE   │
    │  • Sign model hash  │
    │  • Commit to chain  │
    │  • Serve downloads  │
    └─────────────────────┘
```

**Key Modules**:

- **`run.py`**: Main orchestration loop
  - Phase synchronization
  - Seed commitment (phase 1)
  - Model commitment (phase 2)
  - Model aggregation and updates

- **`server.py`**: FastAPI server
  - `/submit-checkpoint`: Receive miner submissions
  - `/get-checkpoint`: Serve model downloads
  - Signature verification
  - Rate limiting

- **`evaluator.py`**: Evaluation engine
  - Load miner checkpoints
  - Run validation set
  - Compute scores (loss, accuracy)
  - Unpredictable validation (prevents gaming)

---

### 2. Miners

**Core Components**:
```
mycelia/miner/
├── model_io.py         # Main miner orchestration
├── train.py            # Training logic
└── train_helper.py     # Training utilities
```

**Miner Flow**:
```
┌────────────────────────────────────────┐
│  Miner System (miner/model_io.py)     │
└──────────┬─────────────────────────────┘
           │
    ┌──────▼──────┐
    │  Scheduler  │  (scheduler_service)
    │  Monitors   │
    │  phases     │
    └──────┬──────┘
           │
    ┌──────▼──────────────────┐
    │  Worker Threads         │
    ├─────────────────────────┤
    │ Download Worker:        │
    │  • Check consensus      │
    │  • Download if agreed   │
    │                         │
    │ Commit Worker:          │
    │  • Compute model hash   │
    │  • Commit to chain      │
    │                         │
    │ Submit Worker:          │
    │  • Find validator       │
    │  • Upload checkpoint    │
    └─────────────────────────┘
```

**Key Modules**:

- **`model_io.py`**: Orchestration with worker threads
  - `scheduler_service()`: Phase monitoring
  - `download_worker()`: Fork-safe model downloads
  - `commit_worker()`: Hash computation and commitment
  - `submit_worker()`: Checkpoint submission

- **`train.py`**: Training loop
  - Expert-specific training
  - Gradient clipping and NaN protection
  - Checkpoint saving
  - Memory management

---

### 3. Shared Infrastructure

**Core Components**:
```
mycelia/shared/
├── chain.py                    # Blockchain interaction
├── cycle.py                    # Phase computation
├── config.py                   # Configuration management
├── fork_safe_download.py       # Consensus-based downloads
├── signature_utils.py          # Cryptographic signatures
├── circuit_breaker.py          # Fault tolerance
├── config_validation.py        # Consensus validation
├── expert_manager.py           # Expert assignment
└── modeling/
    ├── mycelia.py              # MoE model definition
    └── custom_qwen3_vl_moe.py  # Qwen3-VL MoE implementation
```

**Key Modules**:

- **`chain.py`**: Blockchain operations
  - `commit_status()`: Write commits to chain
  - `read_commitment()`: Read commits from chain
  - `get_active_validators()`: Compute active validator set
  - `verify_validator_consensus_on_model()`: Check consensus
  - `get_seed_commit_snapshot_block_hash()`: Deterministic snapshot timing

- **`cycle.py`**: Decentralized phase computation
  - `get_phase()`: Compute current phase from block height
  - `get_validator_miner_assignment()`: Deterministic assignment
  - No central service required - all local computation

- **`fork_safe_download.py`**: Consensus-verified downloads
  - `fetch_model_from_chain_safe()`: Main download function
  - `fetch_model_with_consensus()`: Verify consensus
  - `download_from_validator()`: Streaming download with hash verification

- **`signature_utils.py`**: Cryptographic security
  - `sign_data()`: Sign assignments/messages
  - `verify_signature()`: Verify signatures
  - `verify_assignment_signature()`: Replay attack prevention with timestamps
  - `verify_message_with_nonce()`: Nonce-based verification

- **`circuit_breaker.py`**: Fault tolerance
  - `CircuitBreaker`: Prevent cascading failures
  - `RetryWithBackoff`: Exponential backoff retries
  - `with_timeout()`: Timeout protection

---

## Data Flow

### Model Download Flow

```
┌─────────────┐
│    Miner    │
└──────┬──────┘
       │ 1. Check validator commits on chain
       ▼
┌─────────────────────────────────┐
│  Bittensor Chain                │
│  Read: validator commits         │
│  - Phase 2 commits (model hash) │
│  - Signatures                   │
└──────┬──────────────────────────┘
       │ 2. Verify consensus
       ▼
┌─────────────────────────────────┐
│  fork_safe_download.py          │
│  • ≥60% validators committed?   │
│  • >50% agree on same hash?     │
│  • Verify signatures valid?     │
└──────┬──────────────────────────┘
       │ 3. If consensus achieved
       ▼
┌─────────────────────────────────┐
│  Download from Validator        │
│  • Streaming HTTP download      │
│  • Incremental hash verification│
│  • Progress tracking            │
└──────┬──────────────────────────┘
       │ 4. Verify final hash matches consensus
       ▼
┌─────────────┐
│  Training   │
└─────────────┘
```

### Model Submission Flow

```
┌─────────────┐
│    Miner    │
│  Completes  │
│  Training   │
└──────┬──────┘
       │ 1. Compute checkpoint hash
       ▼
┌─────────────────────────────────┐
│  Commit Hash to Chain           │
│  (COMMIT PHASE)                 │
└──────┬──────────────────────────┘
       │ 2. Find assigned validator
       ▼
┌─────────────────────────────────┐
│  get_validator_miner_assignment │
│  Deterministic assignment from  │
│  seed committed by validators   │
└──────┬──────────────────────────┘
       │ 3. Submit checkpoint
       ▼
┌─────────────────────────────────┐
│  Validator HTTP Server          │
│  POST /submit-checkpoint        │
│  • Verify signature             │
│  • Save checkpoint              │
└──────┬──────────────────────────┘
       │ 4. Evaluate during VALIDATE phase
       ▼
┌─────────────────────────────────┐
│  Validator Evaluator            │
│  • Load checkpoint              │
│  • Run validation set           │
│  • Compute score                │
└──────┬──────────────────────────┘
       │ 5. Aggregate in MERGE phase
       ▼
┌─────────────────────────────────┐
│  Model Aggregation              │
│  • Merge top-K miner updates    │
│  • Update global model          │
│  • Create new checkpoint        │
└─────────────────────────────────┘
```

---

## Consensus Mechanism

### Two-Phase Commit

**Phase 1: Seed Commit**
```
Purpose: Synchronize miner assignments

Validator commits:
{
  "commit_phase": 1,
  "miner_seed": deterministic_seed,
  "expert_group": expert_group_id,
  "global_ver": version,
  "block": current_block
}

Seed computation:
  anchor_block = (current_block // cycle_length) * cycle_length
  anchor_hash = chain.get_block_hash(anchor_block)
  seed = int(anchor_hash[:16], 16)
```

**Phase 2: Model Commit**
```
Purpose: Achieve consensus on model hash

Validator commits:
{
  "commit_phase": 2,
  "model_hash": SHA256(checkpoint),
  "signed_model_hash": signature,
  "expert_group": expert_group_id,
  "global_ver": version,
  "block": current_block
}

Signature format (domain-separated):
  "mycelia:v1|{netuid}|{expert_group}|{anchor_hash}|2|{model_hash}"

Prevents:
  - Replay attacks (includes anchor_hash)
  - Cross-phase reuse (includes phase=2)
  - Cross-network reuse (includes netuid)
```

### Consensus Requirements

```python
def verify_validator_consensus_on_model(
    subtensor,
    netuid,
    expert_group_id,
    block,
    cycle_length,
    phase_periods,
    min_quorum_ratio=0.6,
    min_consensus_ratio=0.5,
):
    # 1. Get snapshot block (END of commit window)
    snapshot_block_hash = get_seed_commit_snapshot_block_hash(...)

    # 2. Get active validators (validator_permit + phase 1 commit)
    active_validators = get_active_validators(
        subtensor, netuid, expert_group_id, snapshot_block_hash
    )

    # 3. Get phase 2 commits (model hashes)
    phase2_commits = [
        commit for commit in get_chain_commits(...)
        if commit.get('p') == 2
    ]

    # 4. Check quorum
    quorum_met = len(phase2_commits) >= len(active_validators) * min_quorum_ratio

    # 5. Find majority hash
    hash_counts = Counter([commit['h'] for commit in phase2_commits])
    consensus_hash, count = hash_counts.most_common(1)[0]

    # 6. Check majority
    majority_met = count >= len(active_validators) * min_consensus_ratio

    # 7. Verify signatures
    for commit in phase2_commits:
        if commit['h'] == consensus_hash:
            verify_model_commit_signature(commit, ...)

    return consensus_hash if (quorum_met and majority_met) else None
```

---

## Security Architecture

### Threat Model

**Threats Prevented**:

1. **Fork Attacks**: Miners download different models
   - **Defense**: Consensus requirement (≥60% quorum + >50% majority)

2. **Replay Attacks**: Reuse old signatures
   - **Defense**: Timestamp validation (5-minute expiry), nonce tracking

3. **Sybil Attacks**: Fake validator identities
   - **Defense**: Validator permit requirement, stake-weighted

4. **Model Tampering**: Modify model in transit
   - **Defense**: Cryptographic signatures, hash verification

5. **Training Gaming**: Submit pre-trained models
   - **Defense**: Unpredictable validation sets, rotation

### Security Layers

```
┌─────────────────────────────────────────┐
│  Layer 1: Blockchain Security          │
│  • Validator permits                   │
│  • Stake requirements                  │
│  • Immutable commit history            │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Layer 2: Cryptographic Security       │
│  • Domain-separated signatures         │
│  • Replay attack prevention            │
│  • Hash verification                   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Layer 3: Consensus Security           │
│  • Quorum + majority requirements      │
│  • Fork-safe downloads                 │
│  • Fail-closed on verification errors  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Layer 4: Application Security         │
│  • Unpredictable validation            │
│  • Checkpoint compatibility checks     │
│  • Rate limiting                       │
└─────────────────────────────────────────┘
```

---

## Performance Characteristics

### Validator Performance

```
Hardware: RTX 4090 (24GB), 64GB RAM, NVMe SSD
Model: Qwen2.5-7B MoE

Evaluation throughput:
  - 100 validation samples/miner
  - ~5 minutes per miner
  - 10 miners/cycle = ~50 minutes

Aggregation:
  - Merge 10 miner checkpoints
  - ~10 minutes

Total cycle time: ~2 hours (600 blocks on mainnet)
```

### Miner Performance

```
Hardware: RTX 3090 (24GB), 32GB RAM

Training throughput:
  - Batch size: 4
  - Gradient accumulation: 4
  - Effective batch size: 16
  - ~200 blocks training time
  - ~2000 training steps

Memory usage:
  - Full model: 28GB (loaded)
  - Active training: 12GB (assigned experts only)
  - Checkpoints: ~7GB per checkpoint
```

---

## Scalability

### Horizontal Scaling

```
Add more miners:
  - Each miner trains different experts
  - Linear scalability up to 64 miners per group
  - Can run multiple expert groups in parallel

Add more validators:
  - Consensus becomes more robust
  - Better geographic distribution
  - Higher fault tolerance
```

### Vertical Scaling

```
Larger models:
  - Scale to Qwen2.5-14B, 32B, 70B
  - More experts per layer
  - Same architecture, more parameters
```

---

## Monitoring & Observability

### Key Metrics

**Validators**:
- Consensus achievement rate
- Miner evaluation throughput
- Model aggregation time
- Network participation rate

**Miners**:
- Download success rate (consensus achieved)
- Training loss convergence
- Submission success rate
- Validation scores

### Logging

All components use structured logging (`structlog`):
```python
logger.info(
    "Model downloaded with consensus",
    consensus_hash=hash[:16],
    validators_agreeing=count,
    size_gb=size,
)
```

---

## Future Improvements

### Planned Features

1. **Resume-able Downloads**: Continue partial downloads on network failure
2. **Compression**: Model compression for faster transfers
3. **Adaptive Batch Sizes**: Auto-tune based on hardware
4. **Multi-Group Training**: Miners train multiple expert groups
5. **Federated Averaging**: Alternative aggregation strategies

---

## Summary

Mycelia's architecture enables decentralized MoE training through:

1. ✅ **Decentralized Coordination**: Blockchain-based consensus
2. ✅ **Fork-Safe Operations**: Consensus before critical operations
3. ✅ **Modular Design**: Clear separation of concerns
4. ✅ **Security-First**: Multiple defensive layers
5. ✅ **Scalable**: Horizontal and vertical scaling supported

The system is production-ready and fully operational.
