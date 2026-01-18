# Metrics API Reference

Complete API documentation for expert-specific evaluation metrics.

## Module: `mycelia.shared.expert_metrics`

### Overview

This module provides specialized metrics for evaluating each expert group:
- **Math Expert (Group 0):** Exact match, numerical accuracy
- **Agentic Expert (Group 1):** Tool call accuracy, function format validation
- **Planning Expert (Group 2):** Step completion, uncertainty calibration, checkpoint usage
- **Vision Expert (Group 3):** VQA accuracy, caption similarity

## Math Expert Metrics

### `math_exact_match()`

Exact string match for mathematical answers.

```python
def math_exact_match(
    predictions: List[str],
    references: List[str]
) -> float
```

**Parameters:**
- `predictions` - Model predictions (extracted answers)
- `references` - Ground truth answers

**Returns:**
- `float` - Accuracy in range [0.0, 1.0]

**Example:**

```python
preds = ["42", "3.14", "x = 5"]
refs = ["42", "3.14159", "x=5"]

accuracy = math_exact_match(preds, refs)
# Returns: 0.67 (2/3 exact matches)
```

---

### `math_numerical_accuracy()`

Numerical comparison with tolerance for floating-point answers.

```python
def math_numerical_accuracy(
    predictions: List[str],
    references: List[str],
    tolerance: float = 1e-3
) -> float
```

**Parameters:**
- `predictions` - Model predictions (must be parseable as numbers)
- `references` - Ground truth numerical answers
- `tolerance` - Acceptable difference (default: 0.001)

**Returns:**
- `float` - Accuracy in range [0.0, 1.0]

**Example:**

```python
preds = ["3.14159", "2.0", "invalid"]
refs = ["3.14", "2.00001", "5"]

accuracy = math_numerical_accuracy(preds, refs, tolerance=0.01)
# Returns: 1.0 (3.14159 ≈ 3.14, 2.0 ≈ 2.00001, "invalid" skipped)
```

---

### `extract_math_answer()`

Extract numerical answer from model output.

```python
def extract_math_answer(text: str) -> str
```

**Parameters:**
- `text` - Full model output text

**Returns:**
- `str` - Extracted answer (last number found)

**Example:**

```python
output = "To solve this, we calculate: 5 + 3 = 8. Therefore, the answer is 8."
answer = extract_math_answer(output)
# Returns: "8"

output = "The result is $1,234.56"
answer = extract_math_answer(output)
# Returns: "1234.56"
```

---

## Agentic Expert Metrics

### `tool_call_accuracy()`

Measures correct tool/function selection.

```python
def tool_call_accuracy(
    predictions: List[List[Dict]],
    references: List[List[Dict]]
) -> float
```

**Parameters:**
- `predictions` - Predicted tool calls: `[{"name": "func", "args": {...}}]`
- `references` - Ground truth tool calls

**Returns:**
- `float` - Accuracy in range [0.0, 1.0]

**Example:**

```python
preds = [
    [{"name": "search", "args": {"query": "AI"}}],
    [{"name": "calculate", "args": {"expr": "2+2"}}],
]
refs = [
    [{"name": "search", "args": {"query": "AI"}}],
    [{"name": "compute", "args": {"expr": "2+2"}}],  # Different function
]

accuracy = tool_call_accuracy(preds, refs)
# Returns: 0.5 (1/2 correct function names)
```

---

### `function_format_valid()`

Validates JSON format of function calls.

```python
def function_format_valid(predictions: List[str]) -> float
```

**Parameters:**
- `predictions` - Raw model outputs (should contain JSON tool calls)

**Returns:**
- `float` - Ratio of valid JSON formats in range [0.0, 1.0]

**Example:**

```python
outputs = [
    '{"name": "search", "args": {"q": "test"}}',  # Valid
    '{name: search, args: {}}',                    # Invalid (no quotes)
    '{"name": "calc"}',                            # Valid
]

validity = function_format_valid(outputs)
# Returns: 0.67 (2/3 valid)
```

---

### `extract_tool_calls()`

Extract tool call JSON from model output.

```python
def extract_tool_calls(text: str) -> List[Dict]
```

**Parameters:**
- `text` - Full model output (may contain markdown, code blocks)

**Returns:**
- `List[Dict]` - Extracted tool calls

**Example:**

```python
output = '''
I'll help you with that. Let me search:
```json
{"name": "search", "args": {"query": "Python tutorial"}}
```
'''

calls = extract_tool_calls(output)
# Returns: [{"name": "search", "args": {"query": "Python tutorial"}}]
```

---

## Planning Expert Metrics

### `step_completion_rate()`

Measures what fraction of plan steps were executed.

```python
def step_completion_rate(
    predictions: List[List[str]],
    references: List[List[str]]
) -> float
```

**Parameters:**
- `predictions` - Model-generated plan steps
- `references` - Expected plan steps

**Returns:**
- `float` - Average completion rate in range [0.0, 1.0]

**Example:**

```python
preds = [
    ["Step 1: Research", "Step 2: Outline", "Step 3: Write"],
    ["Step 1: Install", "Step 2: Configure"],
]
refs = [
    ["Research", "Outline", "Write", "Edit"],  # 3/4 completed
    ["Install", "Configure", "Test", "Deploy"],  # 2/4 completed
]

completion = step_completion_rate(preds, refs)
# Returns: 0.625 ((3/4 + 2/4) / 2)
```

---

### `uncertainty_calibration()`

Measures how well confidence scores match actual accuracy.

```python
def uncertainty_calibration(
    predictions: List[str],
    references: List[str],
    confidences: List[float]
) -> float
```

**Parameters:**
- `predictions` - Model predictions
- `references` - Ground truth
- `confidences` - Model confidence scores (0.0-1.0)

**Returns:**
- `float` - Calibration error (lower is better), in range [0.0, 1.0]

**Example:**

```python
preds = ["A", "B", "C", "D"]
refs = ["A", "A", "C", "C"]  # 2/4 correct
confs = [0.9, 0.9, 0.8, 0.6]

calibration_error = uncertainty_calibration(preds, refs, confs)
# Low calibration error: High confidence predictions were mostly correct
```

---

### `checkpoint_usage()`

Tracks whether model used intermediate checkpoints in multi-step plans.

```python
def checkpoint_usage(predictions: List[str]) -> float
```

**Parameters:**
- `predictions` - Model outputs (should contain checkpoint markers)

**Returns:**
- `float` - Ratio of outputs with checkpoints in range [0.0, 1.0]

**Example:**

```python
outputs = [
    "Step 1 done. [CHECKPOINT] Step 2...",  # Has checkpoint
    "Step 1. Step 2. Step 3.",               # No checkpoint
    "Starting [CHECKPOINT] Middle [CHECKPOINT] End",  # Has checkpoints
]

usage = checkpoint_usage(outputs)
# Returns: 0.67 (2/3 used checkpoints)
```

---

## Vision Expert Metrics

### `vqa_accuracy()`

Visual Question Answering accuracy with soft matching.

```python
def vqa_accuracy(
    predictions: List[str],
    references: List[List[str]],
    soft_match: bool = True
) -> float
```

**Parameters:**
- `predictions` - Model answers
- `references` - Ground truth answers (list of acceptable answers per question)
- `soft_match` - Allow partial matches (e.g., "yes" matches "yeah")

**Returns:**
- `float` - Accuracy in range [0.0, 1.0]

**Example:**

```python
preds = ["yes", "red car", "3 people"]
refs = [
    ["yes", "yeah", "correct"],
    ["red car", "car that is red"],
    ["three people", "3 people", "three"],
]

accuracy = vqa_accuracy(preds, refs, soft_match=True)
# Returns: 1.0 (all match)
```

---

### `caption_similarity()`

Image caption similarity using BLEU and CIDEr metrics.

```python
def caption_similarity(
    predictions: List[str],
    references: List[List[str]]
) -> Tuple[float, float]
```

**Parameters:**
- `predictions` - Generated captions
- `references` - Ground truth captions (multiple references per image)

**Returns:**
- `Tuple[float, float]` - (BLEU-4 score, CIDEr score)

**Example:**

```python
preds = [
    "A dog playing in the park",
    "Red car on the street",
]
refs = [
    ["A dog playing in a park", "Dog at the park playing"],
    ["Red car parked on street", "A red car on the road"],
]

bleu, cider = caption_similarity(preds, refs)
# Returns: (0.65, 1.23)  # BLEU-4, CIDEr
```

---

## Composite Metrics

### `compute_composite_score()`

Combines expert-specific metrics into overall score.

```python
def compute_composite_score(
    metrics: Dict[str, float],
    expert_group_id: int
) -> float
```

**Parameters:**
- `metrics` - Dict of metric_name → score
- `expert_group_id` - Expert group (0=Math, 1=Agentic, 2=Planning, 3=Vision)

**Returns:**
- `float` - Weighted composite score in range [0.0, 1.0]

**Weights by Expert Group:**

```python
WEIGHTS = {
    0: {"math_exact_match": 0.6, "math_numerical_accuracy": 0.4},
    1: {"tool_call_accuracy": 0.7, "function_format_valid": 0.3},
    2: {"step_completion_rate": 0.5, "uncertainty_calibration": 0.3, "checkpoint_usage": 0.2},
    3: {"vqa_accuracy": 0.6, "caption_bleu": 0.2, "caption_cider": 0.2},
}
```

**Example:**

```python
metrics = {
    "math_exact_match": 0.85,
    "math_numerical_accuracy": 0.90,
}

score = compute_composite_score(metrics, expert_group_id=0)
# Returns: 0.6*0.85 + 0.4*0.90 = 0.87
```

---

## Usage Examples

### Example 1: Evaluate Math Expert

```python
from mycelia.shared.expert_metrics import (
    extract_math_answer,
    math_exact_match,
    math_numerical_accuracy,
    compute_composite_score,
)

# Generate predictions
outputs = model.generate(math_inputs)
predictions = [extract_math_answer(out) for out in outputs]

# Compute metrics
exact_acc = math_exact_match(predictions, references)
num_acc = math_numerical_accuracy(predictions, references)

# Composite score
metrics = {
    "math_exact_match": exact_acc,
    "math_numerical_accuracy": num_acc,
}
score = compute_composite_score(metrics, expert_group_id=0)

print(f"Math Exact: {exact_acc:.2%}")
print(f"Math Numerical: {num_acc:.2%}")
print(f"Composite: {score:.2%}")
```

### Example 2: Evaluate Vision Expert

```python
from mycelia.shared.expert_metrics import vqa_accuracy, caption_similarity

# VQA evaluation
vqa_preds = ["yes", "red", "3"]
vqa_refs = [["yes", "yeah"], ["red"], ["3", "three"]]
vqa_acc = vqa_accuracy(vqa_preds, vqa_refs)

# Caption evaluation
cap_preds = ["A dog in the park"]
cap_refs = [["A dog playing in a park", "Dog at the park"]]
bleu, cider = caption_similarity(cap_preds, cap_refs)

# Composite
metrics = {
    "vqa_accuracy": vqa_acc,
    "caption_bleu": bleu,
    "caption_cider": cider,
}
score = compute_composite_score(metrics, expert_group_id=3)

print(f"Vision Score: {score:.2%}")
```

### Example 3: Validator Evaluation

```python
from mycelia.shared.evaluate import evaluate_model

metrics = evaluate_model(
    step=global_step,
    model=miner_model,
    dataloader=validation_dataloader,
    device="cuda",
    expert_group_id=config.task.expert_group_id,
    tokenizer=tokenizer,
    compute_generation_metrics=True,  # Enable expert-specific metrics
)

# metrics contains:
# - composite_score
# - expert_specific_metrics (every 5 batches)
# - loss

print(f"Miner Score: {metrics['composite_score']:.3f}")
```

## Custom Metrics

### Adding New Metrics

1. Define metric function:

```python
def custom_metric(predictions: List[str], references: List[str]) -> float:
    """Your custom metric."""
    # Compute score
    return score
```

2. Update weights in `compute_composite_score()`:

```python
WEIGHTS[expert_group_id]["custom_metric"] = 0.15
```

3. Call in evaluation:

```python
custom_score = custom_metric(preds, refs)
metrics["custom_metric"] = custom_score
```

## References

- Implementation: `mycelia/shared/expert_metrics.py`
- Evaluation: `mycelia/shared/evaluate.py`
- [Expert Design](../../expert_groups/EXPERT_DESIGN.md)
- [MoE Architecture](../architecture/moe_architecture.md)
