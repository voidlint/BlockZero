# Custom Dataset Guide

How to create and configure custom datasets for expert group training.

---

## Table of Contents

- [Overview](#overview)
- [Default Dataset Loader](#default-dataset-loader)
- [Creating Custom Datasets](#creating-custom-datasets)
- [Merged Datasets](#merged-datasets)
- [Configuration](#configuration)
- [Examples](#examples)
- [Best Practices](#best-practices)

---

## Overview

The Mycelia subnet allows you to create custom dataset loaders for each expert group. This enables:

- **Domain-specific data** - Load datasets tailored to your expert's specialization
- **Data preprocessing** - Custom tokenization and formatting
- **Merged datasets** - Combine multiple datasets with configurable weights
- **Streaming support** - Handle large datasets without loading into memory

---

## Default Dataset Loader

### `DefaultStreamingTorchDataset`

Base class for dataset loading with streaming support.

**Location:** `mycelia.shared.dataloader`

**Key Methods:**

```python
class DefaultStreamingTorchDataset:
    @staticmethod
    def tokenize_and_format(
        example: dict[str, str],
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int
    ) -> dict[str, list]

    @classmethod
    def get_tokenised_dataset(
        cls,
        config,
        tokenizer: PreTrainedTokenizerBase,
        rank: int | None = None,
        world_size: int | None = None,
        train: bool = True,
        seed: str | None = None,
        fraction: float | None = None,
    )
```

**Default Behavior:**
- Loads dataset from HuggingFace Hub using `config.task.data.dataset_name`
- Tokenizes text using `tokenizer`
- Supports distributed training with rank/world_size sharding
- Streams data to avoid memory issues

---

## Creating Custom Datasets

### Step 1: Create Dataset Class

Create a new Python file in your expert group directory:

**File:** `expert_groups/exp_math/dataset.py`

```python
from typing import Any
from functools import partial

from datasets import load_dataset, interleave_datasets
from transformers import PreTrainedTokenizerBase

from mycelia.shared.dataloader import DefaultStreamingTorchDataset


class StreamingTorchDataset(DefaultStreamingTorchDataset):
    @staticmethod
    def tokenize_and_format(
        example: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
    ) -> dict[str, Any]:
        """
        Custom tokenization and formatting for your domain.

        Args:
            example: Raw dataset row
            tokenizer: HuggingFace tokenizer
            sequence_length: Maximum sequence length

        Returns:
            Dictionary with "input_ids" and "attention_mask"
        """
        # Handle different data formats
        if "messages" in example:
            # Chat format
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif "text" in example:
            # Plain text
            text = example["text"]
        elif "question" in example and "answer" in example:
            # Q&A format (common in math datasets)
            text = f"Q: {example['question']}\nA: {example['answer']}"
        else:
            # Fallback: concatenate all string fields
            text = " ".join(str(v) for v in example.values() if isinstance(v, str))

        # Tokenize
        toks = tokenizer(
            text,
            truncation=True,
            max_length=sequence_length,
            padding="max_length",
            add_special_tokens=False,
        )

        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }
```

---

## Merged Datasets

### Combining Multiple Datasets

For expert groups that need diverse data, you can merge multiple datasets:

```python
class MergedMathDataset(StreamingTorchDataset):
    """
    Loads and merges multiple math datasets from HuggingFace Hub.
    """

    # Define datasets to merge with weights
    MATH_DATASETS = [
        ("openai/gsm8k", "main", 0.5),           # (name, config, weight)
        ("meta-math/MetaMathQA", None, 0.5),
    ]

    @classmethod
    def get_tokenised_dataset(
        cls,
        config,
        tokenizer: PreTrainedTokenizerBase,
        rank: int | None = None,
        world_size: int | None = None,
        train: bool = True,
        seed: str | None = None,
        fraction: float | None = None,
    ):
        """Load and merge multiple math datasets."""
        import structlog
        logger = structlog.get_logger(__name__)

        logger.info("Loading merged math datasets", train=train)

        # Load all datasets
        datasets = []
        probabilities = []

        for dataset_name, config_name, weight in cls.MATH_DATASETS:
            try:
                logger.info("Loading dataset", name=dataset_name, config=config_name)
                ds = load_dataset(
                    dataset_name,
                    config_name,
                    streaming=True,
                    trust_remote_code=True,
                )

                # Select split
                split_name = "train" if train else "test"
                if split_name not in ds:
                    # Try alternative split names
                    if "validation" in ds:
                        split_name = "validation"
                    elif "dev" in ds:
                        split_name = "dev"
                    else:
                        logger.warning(f"Split not found for {dataset_name}, skipping")
                        continue

                datasets.append(ds[split_name])
                probabilities.append(weight)
                logger.info("Loaded dataset", name=dataset_name, split=split_name)
            except Exception as e:
                logger.error("Failed to load dataset", name=dataset_name, error=str(e))
                continue

        if not datasets:
            raise ValueError("No datasets could be loaded")

        # Normalize probabilities
        total_weight = sum(probabilities)
        probabilities = [p / total_weight for p in probabilities]

        logger.info(f"Merging {len(datasets)} datasets", probs=probabilities)

        # Interleave datasets
        merged_dataset = interleave_datasets(
            datasets,
            probabilities=probabilities,
            seed=42,
            stopping_strategy="all_exhausted",
        )

        # Apply tokenization
        tokenize_fn = partial(
            cls.tokenize_and_format,
            tokenizer=tokenizer,
            sequence_length=config.task.data.sequence_length,
        )

        tokenised = merged_dataset.map(
            tokenize_fn,
            remove_columns=merged_dataset.column_names if hasattr(merged_dataset, 'column_names') else None,
        )

        return tokenised


# Alias for backward compatibility
MergedMathDataset = MergedMathDataset
```

---

## Configuration

### Step 2: Configure Expert Group

Update your expert group config:

**File:** `expert_groups/exp_math/config.yaml`

```yaml
data:
  # For single dataset
  dataset_name: "openai/gsm8k"
  data_dir: "main"

  # Point to your custom dataset class
  dataset_class: "expert_groups.exp_math.dataset:MergedMathDataset"

  # Training parameters
  batch_size: 512
  sequence_length: 1024
  per_device_train_batch_size: 2
  world_size: 10
  rank: 1
  vali_fraction: 0.01

expert_group_id: 0
expert_group_name: "exp_math"
```

### Config Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `dataset_name` | HuggingFace dataset name (for default loader) | `"openai/gsm8k"` |
| `data_dir` | Dataset config/subset name | `"main"` or `null` |
| `dataset_class` | Path to custom dataset class | `"expert_groups.exp_math.dataset:MergedMathDataset"` |
| `batch_size` | Global batch size | `512` |
| `sequence_length` | Maximum sequence length | `1024` |
| `per_device_train_batch_size` | Batch size per GPU | `2` |
| `vali_fraction` | Validation split fraction | `0.01` (1%) |

---

## Examples

### Example 1: Math Dataset (GSM8K + MetaMath)

**Use Case:** Train math reasoning experts

```python
# expert_groups/exp_math/dataset.py
from typing import Any
from functools import partial
from datasets import interleave_datasets, load_dataset
from transformers import PreTrainedTokenizerBase
from mycelia.shared.dataloader import DefaultStreamingTorchDataset

class MergedMathDataset(DefaultStreamingTorchDataset):
    MATH_DATASETS = [
        ("openai/gsm8k", "main", 0.5),
        ("meta-math/MetaMathQA", None, 0.5),
    ]

    @classmethod
    def get_tokenised_dataset(cls, config, tokenizer, **kwargs):
        # ... implementation from above ...
        pass
```

**Config:**

```yaml
data:
  dataset_name: "merged_math"  # Ignored when dataset_class is set
  dataset_class: "expert_groups.exp_math.dataset:MergedMathDataset"
  batch_size: 512
  sequence_length: 1024
```

---

### Example 2: Code Dataset

**Use Case:** Train coding experts

```python
# expert_groups/exp_code/dataset.py
class CodeDataset(DefaultStreamingTorchDataset):
    @staticmethod
    def tokenize_and_format(example, tokenizer, sequence_length):
        # Format code with special markers
        if "code" in example and "description" in example:
            text = f"# {example['description']}\n{example['code']}"
        else:
            text = example.get("text", "")

        toks = tokenizer(
            text,
            truncation=True,
            max_length=sequence_length,
            padding="max_length",
        )

        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }
```

**Config:**

```yaml
data:
  dataset_name: "bigcode/the-stack"
  data_dir: "python"
  dataset_class: "expert_groups.exp_code.dataset:CodeDataset"
```

---

### Example 3: Agentic/Tool-Use Dataset

**Use Case:** Train function-calling experts

```python
# expert_groups/exp_agentic/dataset.py
class AgenticDataset(DefaultStreamingTorchDataset):
    AGENTIC_DATASETS = [
        ("glaiveai/glaive-function-calling-v2", None, 0.5),
        ("Salesforce/xlam-function-calling-60k", None, 0.5),
    ]

    @staticmethod
    def tokenize_and_format(example, tokenizer, sequence_length):
        # Use chat template for function calling
        if "messages" in example:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = example.get("text", "")

        toks = tokenizer(text, truncation=True, max_length=sequence_length, padding="max_length")
        return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}
```

---

## Best Practices

### 1. **Use Streaming for Large Datasets**

✅ **Good:**
```python
ds = load_dataset("large/dataset", streaming=True)
```

❌ **Bad:**
```python
ds = load_dataset("large/dataset", streaming=False)  # Loads entire dataset into RAM
```

---

### 2. **Handle Multiple Data Formats**

✅ **Good:**
```python
def tokenize_and_format(example, tokenizer, sequence_length):
    if "messages" in example:
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    elif "text" in example:
        text = example["text"]
    elif "question" in example:
        text = f"Q: {example['question']}\nA: {example.get('answer', '')}"
    else:
        text = " ".join(str(v) for v in example.values() if isinstance(v, str))
    # ... tokenize ...
```

---

### 3. **Normalize Dataset Weights**

✅ **Good:**
```python
probabilities = [w for _, _, w in DATASETS]
total = sum(probabilities)
probabilities = [p / total for p in probabilities]  # Normalize to sum to 1.0
```

---

### 4. **Graceful Dataset Loading**

✅ **Good:**
```python
for dataset_name, config_name, weight in DATASETS:
    try:
        ds = load_dataset(dataset_name, config_name, streaming=True)
        datasets.append(ds["train"])
    except Exception as e:
        logger.error("Failed to load dataset", name=dataset_name, error=str(e))
        continue  # Skip failed datasets, continue with others
```

---

### 5. **Consistent Sequence Length**

✅ **Good:**
```python
toks = tokenizer(
    text,
    truncation=True,
    max_length=sequence_length,  # Use config value
    padding="max_length",
)
```

---

## Dataset Weights Explained

When merging datasets, weights control sampling probability:

```python
DATASETS = [
    ("dataset_A", None, 0.5),  # 50% of batches from A
    ("dataset_B", None, 0.3),  # 30% of batches from B
    ("dataset_C", None, 0.2),  # 20% of batches from C
]
```

**Use cases:**
- **Equal weights** (`0.33, 0.33, 0.33`) - Balanced sampling
- **Emphasize quality** (`0.7, 0.2, 0.1`) - More samples from high-quality dataset
- **Curriculum learning** - Start with simple, gradually increase complex datasets

---

## Troubleshooting

### Dataset Not Found

```
DatasetNotFoundError: Dataset 'my/dataset' doesn't exist on the Hub
```

**Solution:**
- Verify dataset name on [HuggingFace Hub](https://huggingface.co/datasets)
- Check for typos: `openai/gsm8k` not `openai/gsm-8k`
- Try: `load_dataset("openai/gsm8k", trust_remote_code=True)`

---

### Split Not Found

```
ValueError: Dataset split 'train' not found
```

**Solution:**
```python
split_name = "train" if train else "test"
if split_name not in ds:
    if "validation" in ds:
        split_name = "validation"
    # ... try alternatives ...
```

---

### Out of Memory

**Problem:** Dataset too large for RAM

**Solution:** Use streaming
```python
ds = load_dataset("large/dataset", streaming=True)  # ✅
```

---

## References

- Implementation: `mycelia/shared/dataloader.py`
- Example: `expert_groups/exp_math/dataset.py`
- HuggingFace Datasets: https://huggingface.co/docs/datasets
- Related: [Miner Setup Guide](./miner_setup.md)
