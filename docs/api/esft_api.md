# ESFT API Reference

Complete API documentation for Expert-Specialized Fine-Tuning (ESFT) functions.

## Module: `mycelia.shared.esft_selector`

### Classes

#### `ESFTConfig`

Configuration dataclass for ESFT expert selection.

```python
@dataclass
class ESFTConfig:
    method: Literal["gate", "token"] = "token"
    threshold: float = 0.2
    num_samples: int = 32
    sequence_length: int = 4096
    save_selection: bool = False
    selection_path: Path | None = None
    allow_miner_execution: bool = False
```

**Parameters:**

- `method` - Profiling method:
  - `"gate"`: Gate/router weight-based (threshold ~0.1)
  - `"token"`: Token selection frequency-based (threshold ~0.2, recommended)
- `threshold` - Cumulative score threshold for expert selection (0.0-1.0)
- `num_samples` - Number of samples to profile (default: 32)
- `sequence_length` - Max tokens per sample (default: 4096)
- `save_selection` - Whether to save expert selection to JSON
- `selection_path` - Path to save selection JSON (if `save_selection=True`)
- `allow_miner_execution` - Security flag (default: False, blocks miners)

**Example:**

```python
config = ESFTConfig(
    method="token",
    threshold=0.2,
    num_samples=32,
    save_selection=True,
    selection_path=Path("expert_groups/exp_math/expert_assignment.json"),
)
```

---

#### `ESFTExpertSelector`

Main class for expert profiling and selection.

```python
class ESFTExpertSelector:
    def __init__(self, model: nn.Module, config: ESFTConfig)
```

**Methods:**

##### `accumulate_expert_stats(batch: dict) -> None`

Accumulate expert usage statistics from a batch.

```python
selector = ESFTExpertSelector(model, config)

for batch in sample_dataloader:
    selector.accumulate_expert_stats(batch)
```

**Parameters:**
- `batch` - Dict containing `input_ids`, `attention_mask`, `labels`

**Side Effects:**
- Updates internal expert score accumulators
- Requires model in evaluation mode

---

##### `select_experts() -> Dict[int, List[int]]`

Select experts based on accumulated statistics.

```python
selected_experts = selector.select_experts()

# Returns:
# {
#     0: [0, 2, 5],        # Layer 0: experts 0, 2, 5 selected
#     1: [1, 2, 4, 6],     # Layer 1: experts 1, 2, 4, 6 selected
#     ...
# }
```

**Returns:**
- `Dict[int, List[int]]` - Mapping of layer_id → list of selected expert IDs

**Raises:**
- `RuntimeError` - If called before `accumulate_expert_stats()`

---

##### `freeze_non_selected_experts(selected_experts: Dict[int, List[int]]) -> None`

Freeze all model parameters except selected experts.

```python
selector.freeze_non_selected_experts(selected_experts)

# Result:
# - Selected experts: requires_grad=True
# - All other parameters: requires_grad=False
```

**Parameters:**
- `selected_experts` - Expert selection from `select_experts()`

**Side Effects:**
- Modifies `requires_grad` flags on model parameters
- Prints trainable parameter count

---

### Functions

#### `run_esft_expert_selection()`

High-level wrapper to run complete ESFT workflow.

```python
def run_esft_expert_selection(
    model: nn.Module,
    dataloader: DataLoader,
    config: ESFTConfig,
    device: str = "cuda",
) -> Dict[int, List[int]]
```

**Parameters:**
- `model` - MoE model to profile
- `dataloader` - Dataloader with sample data (should have `config.num_samples` batches)
- `config` - ESFT configuration
- `device` - Device to run on (default: "cuda")

**Returns:**
- `Dict[int, List[int]]` - Selected experts per layer

**Raises:**
- `PermissionError` - If `MYCELIA_ROLE=miner` and `allow_miner_execution=False`

**Example:**

```python
from mycelia.shared.esft_selector import run_esft_expert_selection, ESFTConfig

config = ESFTConfig(method="token", threshold=0.2, num_samples=32)

selected = run_esft_expert_selection(
    model=model,
    dataloader=sample_dataloader,
    config=config,
)

print(f"Selected {sum(len(e) for e in selected.values())} experts")
```

---

#### `load_expert_assignment()`

Load expert selection from JSON file.

```python
def load_expert_assignment(
    assignment_path: Path | str
) -> Dict[int, List[Tuple[int, int]]]
```

**Parameters:**
- `assignment_path` - Path to `expert_assignment.json`

**Returns:**
- `Dict[int, List[Tuple[int, int]]]` - Layer ID → list of (my_expert_id, org_expert_id) tuples

**Example:**

```python
from mycelia.shared.esft_selector import load_expert_assignment

assignment = load_expert_assignment("expert_groups/exp_math/expert_assignment.json")

# Returns:
# {
#     0: [(0, 0), (1, 1)],  # Layer 0: experts (0→0), (1→1)
#     1: [(0, 0), (1, 1)],  # Layer 1: experts (0→0), (1→1)
#     ...
# }
```

---

#### `apply_expert_assignment()`

Apply expert assignment to model (freeze non-assigned experts).

```python
def apply_expert_assignment(
    model: nn.Module,
    assignment: Dict[int, List[Tuple[int, int]]]
) -> None
```

**Parameters:**
- `model` - MoE model
- `assignment` - Expert assignment from `load_expert_assignment()`

**Side Effects:**
- Sets `requires_grad=False` for non-assigned experts
- Sets `requires_grad=True` for assigned experts

**Example:**

```python
from mycelia.shared.esft_selector import load_expert_assignment, apply_expert_assignment

assignment = load_expert_assignment("expert_groups/exp_math/expert_assignment.json")
apply_expert_assignment(model, assignment)

# Now ready to train only assigned experts
```

---

## Usage Examples

### Example 1: Subnet Owner - Run ESFT for Task

```python
from mycelia.shared.esft_selector import run_esft_expert_selection, ESFTConfig
from mycelia.shared.model import get_model
from mycelia.shared.dataloader import get_dataloader
from mycelia.shared.config import MinerConfig

# Load config
config = MinerConfig.from_path("expert_groups/exp_math/config.yaml")
config.task.expert_group_id = 0  # Math

# Load model
model = get_model(config)

# Create sample dataloader (32 samples)
sample_dataloader = get_dataloader(
    config,
    expert_group_id=0,
    tokenizer=tokenizer,
    train=True,
)
# Take only first 32 batches
from itertools import islice
sample_dataloader = islice(sample_dataloader, 32)

# Run ESFT
esft_config = ESFTConfig(
    method="token",
    threshold=0.2,
    num_samples=32,
    save_selection=True,
    selection_path=Path("expert_groups/exp_math/expert_assignment.json"),
)

selected = run_esft_expert_selection(
    model=model,
    dataloader=sample_dataloader,
    config=esft_config,
)

print(f"✓ Selected {sum(len(e) for e in selected.values())} experts")
print(f"✓ Saved to {esft_config.selection_path}")
```

### Example 2: Miner - Load Published Assignment

```python
from mycelia.shared.expert_manager import ExpertManager
from mycelia.shared.model import get_model

# ExpertManager automatically fetches assignment from SN owner
expert_manager = ExpertManager(config)

# Load model with only assigned experts
model = get_model(config, expert_manager=expert_manager)

# Train (only assigned experts have gradients)
for batch in train_dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

### Example 3: Manual Expert Selection

```python
from mycelia.shared.esft_selector import ESFTExpertSelector, ESFTConfig

selector = ESFTExpertSelector(model, config)

# Profile on sample data
model.eval()
with torch.no_grad():
    for i, batch in enumerate(sample_dataloader):
        if i >= 32:  # Limit to 32 samples
            break
        selector.accumulate_expert_stats(batch)

# Select experts
selected_experts = selector.select_experts()

# Analyze selection
for layer_id, expert_ids in selected_experts.items():
    print(f"Layer {layer_id}: Selected {len(expert_ids)}/4 experts → {expert_ids}")

# Freeze non-selected
selector.freeze_non_selected_experts(selected_experts)

# Check trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable/1e9:.2f}B / {total/1e9:.2f}B ({trainable/total:.1%})")
```

## Security Notes

### Miner Execution Block

Miners **cannot** run ESFT:

```python
# In miner training script:
os.environ['MYCELIA_ROLE'] = 'miner'

# Later, if miner tries to run ESFT:
try:
    run_esft_expert_selection(model, dataloader, config)
except PermissionError as e:
    print(e)  # "ESFT selection cannot be run by miners in production"
```

To bypass (for testing only):

```python
config = ESFTConfig(allow_miner_execution=True)  # ⚠️ Testing only!
```

## Best Practices

1. **Sample Size:** Use 32 samples (paper recommendation)
2. **Method:** Prefer `"token"` over `"gate"` (more stable)
3. **Threshold:** Start with 0.2 for token method, 0.1 for gate method
4. **Data Diversity:** Sample from multiple datasets in the expert group
5. **Sequence Length:** Match training sequence length

## References

- [ESFT Overview](../architecture/esft_overview.md)
- [ESFT Security](../ESFT_SECURITY.md)
- [Research Paper](https://arxiv.org/abs/2407.01906)
- Implementation: `mycelia/shared/esft_selector.py`
