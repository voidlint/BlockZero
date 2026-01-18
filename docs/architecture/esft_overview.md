# Expert-Specialized Fine-Tuning (ESFT) Overview

## Introduction

**ESFT (Expert-Specialized Fine-Tuning)** is a methodology for training Mixture-of-Experts models efficiently by selecting and training only task-relevant experts, rather than all experts.

Based on the paper: ["Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models"](https://arxiv.org/abs/2407.01906)

## Key Findings from Research

### 1. Task-Specific Expert Specialization

**Finding:** Each task uses a concentrated subset of experts

```
Math Task:
  Layer 0: Experts [2, 5, 7] activate 80% of the time
  Layer 1: Experts [1, 4, 6] activate 75% of the time
  ...
  Other experts: <20% activation

Conclusion: Only ~30-40% of experts are task-relevant
```

### 2. Cross-Task Differentiation

**Finding:** Different tasks use different expert subsets

```
         Expert 0  Expert 1  Expert 2  Expert 3
Math        ███      ░░░       ███      ░░░
Agentic     ░░░      ███       ░░░      ███
Planning    ███      ░░░       ░░░      ███

Legend: ███ = High usage, ░░░ = Low usage
```

### 3. Training Efficiency

**Finding:** Training only relevant experts matches full fine-tuning (FFT) performance

| Method | Trainable Params | Accuracy | Training Time |
|--------|------------------|----------|---------------|
| FFT    | 100%            | 85.2%    | 10 hours     |
| ESFT   | 30%             | 85.1%    | 3.5 hours    |

**Benefits:**
- 🚀 3x faster training
- 💾 70% less memory
- ✅ Same accuracy
- 🎯 Better specialization

## ESFT Methodology

### Step 1: Expert Profiling

Profile expert usage on a small sample of task data:

```python
from mycelia.shared.esft_selector import ESFTExpertSelector, ESFTConfig

config = ESFTConfig(
    method="token",        # or "gate"
    threshold=0.2,         # Cumulative threshold
    num_samples=32,        # Sample size
    sequence_length=4096,  # Tokens per sample
)

selector = ESFTExpertSelector(model, config)

# Profile on sample data
for batch in sample_dataloader:  # 32 samples
    selector.accumulate_expert_stats(batch)
```

**Two Profiling Methods:**

#### Method 1: Gate-Based (ESFT-Gate)

Measures average gate/routing weights:

```python
# For each expert in each layer:
R_i^l = (1/T) * Σ_t g_i^l(x_t)

Where:
- R_i^l = Relevance score for expert i in layer l
- T = Total tokens
- g_i^l(x_t) = Gate weight for expert i on token t
```

#### Method 2: Token-Based (ESFT-Token)

Measures selection frequency:

```python
# For each expert in each layer:
R_i^l = (1/(T*K)) * Σ_t 1[expert_i selected for token_t]

Where:
- K = top-k (number of experts selected per token)
- 1[...] = Indicator function (1 if selected, 0 otherwise)
```

**Recommendation:** Token-based is more stable (paper finding)

### Step 2: Expert Selection

Select experts using cumulative threshold:

```python
selected_experts = selector.select_experts()

# Example output:
{
    0: [0, 2, 5],        # Layer 0: Experts 0, 2, 5 selected
    1: [1, 2, 4, 6],     # Layer 1: Experts 1, 2, 4, 6 selected
    2: [0, 3, 5],        # Layer 2: Experts 0, 3, 5 selected
    ...
}
```

**Selection Algorithm:**

```python
def select_experts_for_layer(expert_scores, threshold=0.2):
    # Sort experts by relevance score (descending)
    sorted_experts = sorted(expert_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Normalize scores to sum to 1
    total_score = sum(score for _, score in sorted_experts)
    cumulative = 0.0
    selected = []
    
    # Select until cumulative score >= threshold
    for expert_id, score in sorted_experts:
        selected.append(expert_id)
        cumulative += score / total_score
        if cumulative >= threshold:
            break
    
    return selected
```

**Threshold Guidelines:**
- `p=0.1` for gate-based
- `p=0.2` for token-based (recommended)
- Higher threshold = more experts selected

### Step 3: Parameter Freezing

Freeze non-selected experts and other components:

```python
selector.freeze_non_selected_experts(selected_experts)

# What gets frozen/trained:
# ✓ Selected experts: TRAINABLE
# ✗ Non-selected experts: FROZEN
# ✗ Shared expert: FROZEN (degrades general ability if trained)
# ✗ Router/gates: FROZEN (minimal benefit if trained)
# ✗ Attention layers: FROZEN
# ✗ Embeddings: FROZEN
```

**Trainable Parameter Reduction:**

```
Original model: 30B parameters
Selected experts: ~8B parameters (30% of MoE components)
Total trainable: 8B parameters
Memory for gradients/optimizer: 3x reduction
```

## ESFT in Distributed Setting

### Challenge

In a decentralized subnet with multiple miners:
- Each miner selects experts independently
- Different miners select different experts
- Cannot aggregate/merge incompatible expert updates

### Solution: Centralized ESFT

**Workflow:**

```
1. Subnet Owner runs ESFT once per task
   ↓
2. Publish expert selection to all miners
   ↓
3. All miners train same expert subset
   ↓
4. Validators can aggregate expert updates
```

**Implementation:**

```python
# ==== Subnet Owner ====
config = ESFTConfig(
    method="token",
    threshold=0.2,
    save_selection=True,
    selection_path=Path("expert_groups/exp_math/expert_assignment.json"),
)

selected = run_esft_expert_selection(
    model=model,
    dataloader=sample_math_dataloader,
    config=config,
)

# Publish expert_assignment.json to miners via API

# ==== Miner ====
# Fetch from subnet owner (automatic)
expert_manager = ExpertManager(config)
# ✓ Fetched assignment: experts [0,1,2] per layer

# Train assigned experts
model = load_model(config, expert_manager)
# Only selected experts are loaded and trainable
```

### Security

**Problem:** Miners could run ESFT locally to choose "easy" experts

**Solution:** See [ESFT Security](../ESFT_SECURITY.md)

```python
# Miners blocked from running ESFT
try:
    run_esft_expert_selection(model, dataloader, config)
except PermissionError:
    print("ESFT cannot be run by miners!")
```

## Expert Selection Strategies

### Strategy 1: Threshold-Based (Default)

Select experts until cumulative score reaches threshold:

```python
config = ESFTConfig(threshold=0.2)  # Top 20% of expert capacity
```

**Pros:** Adaptive per layer, based on actual usage  
**Cons:** Variable number of experts per layer

### Strategy 2: Top-K

Select fixed number of experts per layer:

```python
def select_topk_experts(scores, k=2):
    return sorted(scores, key=scores.get, reverse=True)[:k]
```

**Pros:** Predictable memory usage  
**Cons:** May over/under-select for some layers

### Strategy 3: Hybrid

Combination of threshold + max limit:

```python
selected = select_with_threshold(scores, threshold=0.2)
if len(selected) > max_experts:
    selected = selected[:max_experts]  # Cap at maximum
```

## Validation & Results

### Expert Coverage Analysis

```python
from mycelia.shared.expert_profiler import analyze_expert_coverage

coverage = analyze_expert_coverage(
    model=model,
    dataloader=validation_dataloader,
    selected_experts=selected_experts,
)

print(f"Token coverage: {coverage['token_coverage']:.1%}")
# → 92.5% of tokens use selected experts

print(f"Expert utilization: {coverage['expert_utilization']:.1%}")
# → Selected experts handle 95% of routing decisions
```

### Performance Comparison

| Method | Parameters | Math Acc | Code Acc | General Acc | Memory |
|--------|-----------|----------|----------|-------------|--------|
| FFT    | 30B (100%) | 89.2%   | 76.5%   | 71.3%      | 180 GB |
| ESFT   | 9B (30%)   | 89.0%   | 76.3%   | 71.1%      | 60 GB  |
| LoRA   | 0.3B (1%)  | 85.1%   | 72.0%   | 69.8%      | 35 GB  |

**Key Takeaway:** ESFT matches FFT with 3x less resources

## Best Practices

### 1. Sample Size

```python
# Paper recommendation: 32 samples
config = ESFTConfig(num_samples=32)

# Minimum: 16 (faster but less stable)
# Maximum: 128 (more stable but slower profiling)
```

### 2. Sequence Length

```python
# Use representative sequence length
config = ESFTConfig(sequence_length=4096)

# Math: 2048-4096 (long problem solving)
# Chat: 1024-2048 (conversational)
# Code: 4096-8192 (long contexts)
```

### 3. Data Sampling

```python
# Sample from multiple datasets
sample_dataloader = DataLoader(
    ConcatDataset([
        Subset(gsm8k_dataset, range(10)),
        Subset(math_dataset, range(10)),
        Subset(amc_dataset, range(12)),
    ]),
    batch_size=2,
)
```

### 4. Threshold Selection

```python
# Token method (recommended)
config = ESFTConfig(method="token", threshold=0.2)

# For difficult tasks, increase threshold:
config = ESFTConfig(threshold=0.3)  # Use more experts
```

## Implementation Details

### File Structure

```
mycelia/shared/esft_selector.py
├── ESFTConfig              # Configuration dataclass
├── ESFTExpertSelector      # Main selector class
│   ├── accumulate_expert_stats()   # Step 1: Profile
│   ├── select_experts()            # Step 2: Select
│   └── freeze_non_selected()       # Step 3: Freeze
├── run_esft_expert_selection()    # High-level wrapper
├── load_expert_assignment()       # Load published selection
└── apply_expert_assignment()      # Apply to model
```

### Usage Examples

**Example 1: Subnet Owner - Run ESFT**

```python
from mycelia.shared.esft_selector import run_esft_expert_selection, ESFTConfig

config = ESFTConfig(
    method="token",
    threshold=0.2,
    num_samples=32,
    save_selection=True,
    selection_path=Path("expert_groups/exp_math/expert_assignment.json"),
)

selected_experts = run_esft_expert_selection(
    model=model,
    dataloader=sample_dataloader,
    config=config,
)

print(f"Selected {sum(len(e) for e in selected_experts.values())} experts")
# → Selected 156 experts across 48 layers
```

**Example 2: Miner - Load Published Selection**

```python
from mycelia.shared.esft_selector import load_expert_assignment, apply_expert_assignment

# Load subnet owner's published selection
selected_experts = load_expert_assignment("expert_groups/exp_math/expert_assignment.json")

# Apply to model
apply_expert_assignment(model, selected_experts)

# Now train
for batch in train_dataloader:
    loss = model(**batch).loss
    loss.backward()  # Only selected experts have gradients
    optimizer.step()
```

## References

- **Paper:** [arXiv:2407.01906](https://arxiv.org/abs/2407.01906)
- **Implementation:** `mycelia/shared/esft_selector.py`
- **Security:** [ESFT Security](../ESFT_SECURITY.md)
- **API Reference:** [ESFT API](../api/esft_api.md)

## Next Steps

- See [MoE Architecture](moe_architecture.md) for overall system
- See [Subnet Owner Guide](../guides/subnet_owner_guide.md) to run ESFT
- See [API Reference](../api/esft_api.md) for detailed function docs

