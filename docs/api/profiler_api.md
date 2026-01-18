# Profiler API Reference

Complete API documentation for expert profiling and analysis tools.

## Module: `mycelia.sn_owner.analyze_expert_routing`

### Overview

Tools for analyzing expert routing patterns, generating skill matrices, and visualizing expert specialization across the MoE model.

## Main Functions

### `analyze_expert_routing()`

Main entry point for expert routing analysis.

```python
def analyze_expert_routing(
    model: nn.Module,
    dataloader: DataLoader,
    task_name: str,
    output_dir: Path = Path("analysis/"),
    num_batches: int = 10,
    device: str = "cuda"
) -> Dict[str, Any]
```

**Parameters:**
- `model` - MoE model to analyze
- `dataloader` - Dataset to profile on
- `task_name` - Name of task (e.g., "math", "vision")
- `output_dir` - Directory to save visualizations (default: "analysis/")
- `num_batches` - Number of batches to analyze (default: 10)
- `device` - Device to run on (default: "cuda")

**Returns:**
- `Dict[str, Any]` - Analysis results containing:
  - `"expert_usage"` - Per-layer expert activation frequencies
  - `"routing_patterns"` - Token-level routing decisions
  - `"skill_matrix"` - Expert specialization scores
  - `"layer_specialization"` - Layer-wise analysis

**Example:**

```python
from mycelia.sn_owner.analyze_expert_routing import analyze_expert_routing

results = analyze_expert_routing(
    model=model,
    dataloader=math_dataloader,
    task_name="math",
    output_dir=Path("analysis/math"),
    num_batches=20,
)

print(f"Expert usage: {results['expert_usage']}")
# Saves: analysis/math/expert_usage_heatmap.png
#        analysis/math/skill_matrix.png
#        analysis/math/routing_patterns.json
```

---

### `compute_skill_matrix()`

Compute task-specific expert specialization matrix.

```python
def compute_skill_matrix(
    model: nn.Module,
    task_dataloaders: Dict[str, DataLoader],
    num_samples: int = 32,
    device: str = "cuda"
) -> np.ndarray
```

**Parameters:**
- `model` - MoE model
- `task_dataloaders` - Dict of task_name → dataloader
- `num_samples` - Samples per task (default: 32)
- `device` - Device to run on

**Returns:**
- `np.ndarray` - Skill matrix of shape `[num_layers, num_experts, num_tasks]`

**Example:**

```python
from mycelia.sn_owner.analyze_expert_routing import compute_skill_matrix

dataloaders = {
    "math": math_dataloader,
    "code": code_dataloader,
    "chat": chat_dataloader,
}

skill_matrix = compute_skill_matrix(
    model=model,
    task_dataloaders=dataloaders,
    num_samples=32,
)

# skill_matrix[layer_id, expert_id, task_id] = specialization_score
print(f"Layer 0, Expert 2, Math: {skill_matrix[0, 2, 0]:.2f}")
```

---

### `visualize_expert_usage()`

Create heatmap visualization of expert activation patterns.

```python
def visualize_expert_usage(
    expert_usage: np.ndarray,
    output_path: Path,
    title: str = "Expert Usage Heatmap"
) -> None
```

**Parameters:**
- `expert_usage` - Array of shape `[num_layers, num_experts]` with activation frequencies
- `output_path` - Where to save PNG
- `title` - Plot title

**Side Effects:**
- Saves heatmap to `output_path`

**Example:**

```python
import numpy as np
from mycelia.sn_owner.analyze_expert_routing import visualize_expert_usage

# Shape: [48 layers, 4 experts]
expert_usage = np.random.rand(48, 4)

visualize_expert_usage(
    expert_usage=expert_usage,
    output_path=Path("analysis/usage.png"),
    title="Math Task Expert Usage"
)
```

---

### `compare_routing_patterns()`

Compare expert routing across multiple tasks.

```python
def compare_routing_patterns(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    output_dir: Path = Path("analysis/comparison"),
    num_batches: int = 10
) -> Dict[str, np.ndarray]
```

**Parameters:**
- `model` - MoE model
- `dataloaders` - Dict of task_name → dataloader
- `output_dir` - Where to save comparison visualizations
- `num_batches` - Batches per task

**Returns:**
- `Dict[str, np.ndarray]` - Task-specific routing patterns

**Example:**

```python
from mycelia.sn_owner.analyze_expert_routing import compare_routing_patterns

dataloaders = {
    "math": math_dataloader,
    "vision": vision_dataloader,
    "planning": planning_dataloader,
}

patterns = compare_routing_patterns(
    model=model,
    dataloaders=dataloaders,
    output_dir=Path("analysis/comparison"),
)

# Saves: analysis/comparison/task_comparison.png
#        analysis/comparison/expert_overlap.png
```

---

## ExpertProfiler Class

Advanced profiling with per-token tracking.

```python
class ExpertProfiler:
    """Profile expert routing at token level."""
    
    def __init__(self, model: nn.Module, track_tokens: bool = True)
```

### Methods

#### `profile_batch()`

Profile expert routing for a single batch.

```python
def profile_batch(
    self,
    batch: Dict[str, torch.Tensor]
) -> Dict[str, Any]
```

**Parameters:**
- `batch` - Input batch with `input_ids`, `attention_mask`

**Returns:**
- `Dict[str, Any]` - Batch profiling results:
  - `"routing_weights"` - Router probabilities per token
  - `"selected_experts"` - Top-k expert IDs per token
  - `"gate_logits"` - Raw gate outputs
  - `"token_ids"` - Input token IDs (if `track_tokens=True`)

**Example:**

```python
from mycelia.sn_owner.analyze_expert_routing import ExpertProfiler

profiler = ExpertProfiler(model, track_tokens=True)

batch = next(iter(dataloader))
results = profiler.profile_batch(batch)

print(f"Shape: {results['routing_weights'].shape}")
# [batch_size, seq_len, num_layers, num_experts]
```

---

#### `aggregate_statistics()`

Aggregate profiling statistics across multiple batches.

```python
def aggregate_statistics(self) -> Dict[str, np.ndarray]
```

**Returns:**
- `Dict[str, np.ndarray]` - Aggregated statistics:
  - `"mean_usage"` - Average expert usage per layer
  - `"std_usage"` - Standard deviation
  - `"max_usage"` - Maximum activation frequency
  - `"specialization_index"` - Gini coefficient (0=uniform, 1=specialized)

**Example:**

```python
profiler = ExpertProfiler(model)

for batch in dataloader:
    profiler.profile_batch(batch)

stats = profiler.aggregate_statistics()
print(f"Expert specialization: {stats['specialization_index']}")
```

---

#### `get_token_level_routing()`

Get detailed token-level routing decisions.

```python
def get_token_level_routing(
    self,
    tokens: List[str]
) -> pd.DataFrame
```

**Parameters:**
- `tokens` - List of token strings to analyze

**Returns:**
- `pd.DataFrame` - Columns: `token`, `layer`, `top1_expert`, `top2_expert`, `top1_prob`, `top2_prob`

**Example:**

```python
profiler = ExpertProfiler(model, track_tokens=True)
profiler.profile_batch(batch)

routing_df = profiler.get_token_level_routing(
    tokens=["What", "is", "2", "+", "2", "?"]
)

print(routing_df)
#   token  layer  top1_expert  top2_expert  top1_prob  top2_prob
# 0  What      0            0            2      0.65      0.35
# 1  What      1            1            3      0.58      0.42
# 2    is      0            0            1      0.72      0.28
# ...
```

---

## Utility Functions

### `calculate_expert_diversity()`

Measure how diverse expert selection is across layers.

```python
def calculate_expert_diversity(
    routing_patterns: np.ndarray
) -> float
```

**Parameters:**
- `routing_patterns` - Array of shape `[num_layers, num_experts]`

**Returns:**
- `float` - Diversity score (0=all experts equal, 1=one expert dominates)

**Example:**

```python
from mycelia.sn_owner.analyze_expert_routing import calculate_expert_diversity

# Uniform usage
uniform = np.ones((48, 4)) * 0.25
diversity = calculate_expert_diversity(uniform)
# Returns: ~0.0 (perfectly diverse)

# Specialized usage
specialized = np.zeros((48, 4))
specialized[:, 0] = 1.0  # Expert 0 always selected
diversity = calculate_expert_diversity(specialized)
# Returns: 1.0 (not diverse at all)
```

---

### `find_task_specific_experts()`

Identify which experts specialize in specific tasks.

```python
def find_task_specific_experts(
    skill_matrix: np.ndarray,
    task_id: int,
    threshold: float = 0.8
) -> List[Tuple[int, int]]
```

**Parameters:**
- `skill_matrix` - From `compute_skill_matrix()`
- `task_id` - Task index
- `threshold` - Minimum specialization score

**Returns:**
- `List[Tuple[int, int]]` - List of (layer_id, expert_id) specialized in task

**Example:**

```python
from mycelia.sn_owner.analyze_expert_routing import find_task_specific_experts

# Find experts specialized in math (task_id=0)
math_experts = find_task_specific_experts(
    skill_matrix=skill_matrix,
    task_id=0,
    threshold=0.8,
)

print(f"Math-specialized experts: {math_experts}")
# [(0, 0), (0, 1), (1, 0), (1, 1), ...]
```

---

## Complete Example: Profiling Workflow

```python
from pathlib import Path
from mycelia.shared.model import get_model
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.config import MinerConfig
from mycelia.sn_owner.analyze_expert_routing import (
    analyze_expert_routing,
    compute_skill_matrix,
    compare_routing_patterns,
    ExpertProfiler,
)

# 1. Load model and data
config = MinerConfig.from_path("config.yaml")
model = get_model(config)

dataloaders = {
    "math": get_dataloader(config, expert_group_id=0, train=False),
    "agentic": get_dataloader(config, expert_group_id=1, train=False),
    "planning": get_dataloader(config, expert_group_id=2, train=False),
    "vision": get_dataloader(config, expert_group_id=3, train=False),
}

# 2. Analyze each task
for task_name, dataloader in dataloaders.items():
    results = analyze_expert_routing(
        model=model,
        dataloader=dataloader,
        task_name=task_name,
        output_dir=Path(f"analysis/{task_name}"),
        num_batches=20,
    )
    print(f"✓ Analyzed {task_name}")

# 3. Compute cross-task skill matrix
skill_matrix = compute_skill_matrix(
    model=model,
    task_dataloaders=dataloaders,
    num_samples=32,
)
np.save("analysis/skill_matrix.npy", skill_matrix)

# 4. Compare routing patterns
patterns = compare_routing_patterns(
    model=model,
    dataloaders=dataloaders,
    output_dir=Path("analysis/comparison"),
)

# 5. Token-level profiling (optional, detailed)
profiler = ExpertProfiler(model, track_tokens=True)
for i, batch in enumerate(dataloaders["math"]):
    if i >= 10:
        break
    profiler.profile_batch(batch)

stats = profiler.aggregate_statistics()
print(f"Specialization: {stats['specialization_index'].mean():.2f}")
```

## Output Files

Running profiling generates:

```
analysis/
├── math/
│   ├── expert_usage_heatmap.png
│   ├── skill_matrix.png
│   └── routing_patterns.json
├── agentic/
│   └── ...
├── planning/
│   └── ...
├── vision/
│   └── ...
├── comparison/
│   ├── task_comparison.png
│   └── expert_overlap.png
└── skill_matrix.npy
```

## References

- Implementation: `mycelia/sn_owner/analyze_expert_routing.py`
- [MoE Architecture](../architecture/moe_architecture.md)
- [ESFT Overview](../architecture/esft_overview.md)
- [Subnet Owner Guide](../guides/subnet_owner_guide.md)
