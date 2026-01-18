# How Mixture of Experts (MoE) Works

A comprehensive explanation of the Mixture of Experts architecture used in Block Zero.

---

## Table of Contents

- [What is Mixture of Experts?](#what-is-mixture-of-experts)
- [Why MoE for Decentralized Training?](#why-moe-for-decentralized-training)
- [Architecture Overview](#architecture-overview)
- [How Experts are Trained](#how-experts-are-trained)
- [Expert Routing](#expert-routing)
- [Expert Groups in Mycelia](#expert-groups-in-mycelia)
- [Training Process](#training-process)
- [Technical Details](#technical-details)

---

## What is Mixture of Experts?

**Mixture of Experts (MoE)** is a neural network architecture that uses multiple specialized sub-networks (experts) instead of one monolithic model.

### Traditional Model vs MoE

**Traditional Dense Model**:
```
Input → [Huge Single Network] → Output
        All parameters active
        Memory: O(N)
        Compute: O(N)
```

**Mixture of Experts**:
```
Input → Router → [Expert 1] → Weighted
        ↓        [Expert 2]    Combination → Output
        ↓        [Expert 3]
        ↓        [Expert 4]
        └──────→ [Expert N]

Only top-K experts active per input
Memory: O(N)
Compute: O(k) where k << N
```

### Key Advantages

1. **Sparse Activation**: Only a few experts process each input
2. **Specialization**: Different experts learn different skills
3. **Scalability**: Add more experts without increasing compute per input
4. **Parallelization**: Different experts can be trained independently

---

## Why MoE for Decentralized Training?

MoE is **perfect for decentralized training** because:

### 1. Expert Independence

Experts are relatively independent modules that can be:
- Trained separately by different miners
- Updated asynchronously
- Specialized for different tasks/domains

### 2. Natural Task Division

```
Validator: "Miner A, train experts 0,5,12 for Math"
           "Miner B, train experts 1,7,13 for Math"
           "Miner C, train experts 2,8,14 for Math"

Each miner trains their assigned experts independently
Validator aggregates the updates
```

### 3. Efficient Resource Usage

```
Instead of:
  10 miners × 70B parameters = 700B parameter-updates

We have:
  10 miners × 7B parameters each = 70B total model
  But trained in parallel with 10x efficiency
```

### 4. Domain Specialization

Different expert groups can specialize in different domains:
- **Math experts**: Arithmetic, algebra, calculus
- **Vision experts**: Object detection, segmentation, OCR
- **Planning experts**: Strategy, optimization, reasoning

---

## Architecture Overview

### Model Structure

```
┌─────────────────────────────────────────────────┐
│              Qwen3-vl Base Model                │
│                                                 │
│  ┌────────────────────────────────────────┐     │
│  │         Transformer Layers (0-31)      │     │
│  │                                        │     │
│  │  Layer 0:  [Attention] → [MoE FFN]     │     │
│  │            ↓             ↓             │     │
│  │            Router → 64 Experts         │     │
│  │            Selects top-8 per token     │     │
│  │                                        │     │
│  │  Layer 1:  [Attention] → [MoE FFN]     │     │
│  │            ...                         │     │
│  │  Layer 31: [Attention] → [MoE FFN]     │     │
│  └────────────────────────────────────────┘     │
│                                                 │
│  Total: 32 layers × 64 experts = 2048 experts   │
└─────────────────────────────────────────────────┘
```

### Expert Layout

Each transformer layer has:
- **Attention mechanism**: Shared across all inputs
- **MoE Feed-Forward Network**: 64 experts
  - **Router**: Decides which experts to use
  - **Experts**: Specialized feed-forward networks
  - **Top-K**: Activates 8 experts per token

### Parameters

```
Base Model: Qwen2.5-7B
├── Embedding: ~500M parameters
├── 32 Transformer Layers:
│   ├── Attention: ~100M parameters each
│   └── MoE FFN: ~150M parameters each
│       ├── Router: ~10M parameters
│       └── 64 Experts × ~2M each = ~128M
└── Output Head: ~500M parameters

Total: ~7B parameters
Active per token: ~2B parameters (top-8 of 64 experts)
```

---

## How Experts are Trained

### Decentralized Training Flow

```
Cycle N:
┌──────────────────────────────────────────────────┐
│ Validator assigns experts to miners:             │
│                                                  │
│ Miner A → Experts [0, 8, 16, 24] in Layer 0-31   │
│ Miner B → Experts [1, 9, 17, 25] in Layer 0-31   │
│ Miner C → Experts [2, 10, 18, 26] in Layer 0-31  │
│ ...                                              │
│ Miner P → Experts [7, 15, 23, 31] in Layer 0-31  │
└──────────────────────────────────────────────────┘
                      ↓
         Each miner trains their experts
                      ↓
┌──────────────────────────────────────────────────┐
│ Miners submit checkpoints to validator           │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│ Validator evaluates submissions:                 │
│ - Runs validation set                            │
│ - Measures loss/accuracy                         │
│ - Ranks miners by performance                    │
└──────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────┐
│ Validator aggregates top submissions:            │
│ - Merges gradient updates                        │
│ - Updates global model                           │
│ - Creates checkpoint for next cycle              │
└──────────────────────────────────────────────────┘
```

### Parameter Freezing

When training, miners:
1. **Load full model** (all layers, all experts)
2. **Freeze non-assigned experts** (gradients disabled)
3. **Train only assigned experts** (gradients enabled)
4. **Submit updated checkpoint** (only assigned expert weights changed)

```python
# Example: Miner assigned experts [0, 8, 16, 24]
for name, param in model.named_parameters():
    layer_id, expert_id = extract_ids(name)

    if expert_id in [0, 8, 16, 24]:
        param.requires_grad = True   # Train these
    else:
        param.requires_grad = False  # Freeze these
```

---

## Expert Routing

### How the Router Works

For each token in the input:

```
Token → Router Network → Logits for all 64 experts
         ↓
      Softmax → Probabilities
         ↓
      Top-K Selection (K=8)
         ↓
   [Expert 3: 0.25]
   [Expert 7: 0.18]    Selected experts + weights
   [Expert 12: 0.15]
   [Expert 19: 0.12]
   [Expert 24: 0.10]
   [Expert 31: 0.08]
   [Expert 45: 0.07]
   [Expert 52: 0.05]
         ↓
   Weighted combination of expert outputs
```

### Load Balancing

The router includes **load balancing loss** to ensure:
- All experts are used roughly equally
- No expert becomes dominant
- Training is distributed fairly

```python
router_loss = load_balance_loss(expert_usage_counts)
total_loss = task_loss + α * router_loss
```

This ensures miners' assigned experts get used, so their training contributes.

---

## Expert Groups in Mycelia

### Group Specialization

Mycelia organizes experts into **domain-specific groups**:
┌─----------------------------------------------------------------------------------------------┐
| Group ID | Domain |              Example Tasks                  |       Experts Trained       |
|----------|--------|---------------------------------------------|-----------------------------|
| 0        | Math   | Arithmetic, algebra, calculus, proofs       | 64 experts across 32 layers |
| 1        | Agentic| Multi-step reasoning, planning, tool use    | 64 experts across 32 layers |
| 2        |Planning| Strategic decisions, optimization           | 64 experts across 32 layers |
| 3        | Vision | Image understanding, multimodal tasks.      | 64 experts across 32 layers |
| 4        |Robotics| Control, perception, action planning        | 64 experts across 32 layers |
└-----------------------------------------------------------------------------------------------┘

### Why Separate Groups?

**Data Specialization**:
```
Math Group trained on:
- Math word problems
- Equations
- Proofs
- Numerical reasoning

Vision Group trained on:
- Image-text pairs
- Visual question answering
- OCR tasks
- Object detection
```

**Evaluation Specialization**:
```
Math Group evaluated on:
- GSM8K (grade school math)
- MATH dataset (competition math)
- Calculation accuracy

Vision Group evaluated on:
- VQA (visual question answering)
- COCO captions
- Image classification
```

### Expert Assignment within Groups

Within a group, miners are assigned specific experts:

```
Math Group (64 experts per layer, 32 layers):
├── Miner 1: [0, 8, 16, 24, 32, 40, 48, 56] across all layers
├── Miner 2: [1, 9, 17, 25, 33, 41, 49, 57] across all layers
├── Miner 3: [2, 10, 18, 26, 34, 42, 50, 58] across all layers
└── ...
└── Miner 8: [7, 15, 23, 31, 39, 47, 55, 63] across all layers

Total: 64 experts × 32 layers = 2048 expert-layer combinations
Assignment rotates each cycle for fairness
```

---

## Training Process

### 1. Forward Pass

```
Input text: "What is 15 × 23?"
    ↓
Tokenize: [15, ×, 23, ?]
    ↓
For each layer (0-31):
    Attention(tokens)
    ↓
    Router selects top-8 experts for each token
    ↓
    [Token "15" → Experts 3,7,12,19,24,31,45,52]
    [Token "×"  → Experts 1,5,8,15,22,33,41,58]
    [Token "23" → Experts 2,9,11,18,27,35,43,51]
    [Token "?" → Experts 4,6,14,20,28,36,44,60]
    ↓
    Each expert processes its assigned tokens
    ↓
    Weighted combination of expert outputs
    ↓
Next layer...
    ↓
Final output: "345"
```

### 2. Backward Pass

```
Loss = CrossEntropy(output, target)
    ↓
Compute gradients: ∂Loss/∂θ
    ↓
Only update gradients for ASSIGNED experts
├── Miner A's experts: gradients computed ✓
├── Miner B's experts: gradients frozen ✗
└── Miner C's experts: gradients frozen ✗
    ↓
Optimizer step (only on assigned expert parameters)
    ↓
Checkpoint saved
```

### 3. Aggregation

Validator merges updates from all miners:

```
Global Model ← α × Miner_A_updates
             + β × Miner_B_updates
             + γ × Miner_C_updates
             + ...

Where α, β, γ are weights based on:
- Validation performance
- Stake
- Historical contribution
```

---

## Technical Details

### Expert Structure

Each expert is a simple feed-forward network:

```python
class Expert(nn.Module):
    def __init__(self, dim=3584):
        super().__init__()
        self.w1 = nn.Linear(dim, 4 * dim)  # Up projection
        self.w2 = nn.Linear(4 * dim, dim)  # Down projection
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))
```

### Router Structure

```python
class Router(nn.Module):
    def __init__(self, dim=3584, num_experts=64):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.top_k = 8

    def forward(self, x):
        # x: [batch, seq_len, dim]
        logits = self.gate(x)  # [batch, seq_len, num_experts]

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_indices, top_k_weights
```

### MoE Layer

```python
class MoELayer(nn.Module):
    def __init__(self, num_experts=64, top_k=8):
        super().__init__()
        self.router = Router(num_experts=num_experts)
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        # Route tokens to experts
        expert_indices, expert_weights = self.router(x)

        # Compute expert outputs
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            mask = (expert_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x[mask])
                weight = expert_weights[mask, expert_indices[mask] == i]
                output[mask] += weight * expert_out

        return output
```

---

## Advantages for Decentralized Training

### 1. Parallelization

```
Traditional: 10 miners × 7B model = Sequential updates
MoE: 10 miners × 700M experts each = Parallel updates
```

### 2. Specialization

```
Miner with good GPU → Vision experts (computationally intensive)
Miner with good NLP data → Math experts (data-intensive)
```

### 3. Fault Tolerance

```
If Miner A goes offline:
- Other miners continue training their experts
- Expert assignment rotates next cycle
- No single point of failure
```

### 4. Scalability

```
Add more experts → More capacity
Add more miners → More parallelization
No increase in per-token compute
```

---

## Performance Characteristics

### Compute Efficiency

```
Dense 7B Model:
- 7B parameters active per forward pass
- Memory: 28GB (fp32), 14GB (fp16)
- FLOPs: ~14T per token

MoE 7B Model (8 of 64 experts):
- ~2B parameters active per forward pass
- Memory: 28GB total, but 8GB active
- FLOPs: ~4T per token (3.5x faster)
```

### Memory Efficiency

```
Training:
- Full model in memory: 28GB
- Only compute gradients for assigned experts
- Gradient memory: ~500MB per miner (instead of 28GB)
```

---

## Summary

**Mixture of Experts** enables Mycelia's decentralized training by:

1. ✅ **Modular Design**: Experts can be trained independently
2. ✅ **Sparse Activation**: Efficient inference (only top-K active)
3. ✅ **Domain Specialization**: Different expert groups for different tasks
4. ✅ **Parallel Training**: Multiple miners train different experts simultaneously
5. ✅ **Scalability**: Add more experts without increasing per-token cost

This makes it possible for a distributed network to collaboratively train a large model that would be impossible for any single entity to train alone.

---

## Further Reading

- [MoE Original Paper](placeholder)
- [Switch Transformers](placeholder)
- [Qwen3 vl Technical Report](placeholder)
- [Block Zero Architecture Documentation](ARCHITECTURE.md)
