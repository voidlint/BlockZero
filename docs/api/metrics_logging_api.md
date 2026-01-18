# Metrics Logging API Reference

Complete API documentation for the metrics logging system.

## Module: `mycelia.shared.metrics`

### Overview

Provides metrics logging to CSV files and optional Weights & Biases (W&B) integration for training/validation metrics tracking.

---

## Classes

### `MetricLogger`

Handles logging of training/validation metrics to CSV and optionally to Weights & Biases.

```python
class MetricLogger:
    """Handles logging of training/validation metrics."""

    def __init__(self, config: Any, rank: int = 0) -> None
```

**Parameters:**
- `config` - Configuration object containing logging settings (must have `config.log` attributes)
- `rank` (int, optional) - Process rank for distributed training. Only rank 0 logs. Default: 0

**Attributes:**
- `config` - Stored configuration
- `rank` - Process rank
- `metric_path` (Path) - Path to CSV file from `config.log.metric_path`
- `log_wandb` (bool) - Whether to log to W&B from `config.log.log_wandb`
- `wandb_run` - W&B run object (if enabled)

**Raises:**
- `ImportError` - If W&B logging is enabled but wandb is not installed

---

## Methods

### `__init__()`

Initialize the metric logger.

**Example:**

```python
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.config import MinerConfig

config = MinerConfig.from_path("config.yaml")
metric_logger = MetricLogger(config, rank=0)
```

**Initialization Behavior:**
- Creates metrics directory if it doesn't exist
- Initializes W&B if `config.log.log_wandb` is `True`
- Only rank 0 performs actual logging (distributed training safe)

---

### `log()`

Log metrics to CSV and optionally to W&B.

```python
def log(self, metrics: dict[str, Any], print_log: bool = True) -> None
```

**Parameters:**
- `metrics` (dict) - Dictionary of metric names and values
- `print_log` (bool, optional) - Whether to print metrics to logger. Default: `True`

**Returns:**
- `None`

**Side Effects:**
- Appends metrics to CSV file at `config.log.metric_path`
- Logs to W&B if enabled
- Prints to structured logger if `print_log=True`

**Features:**
- **Dynamic fieldnames** - Automatically handles new metric keys across calls
- **Rank-aware** - Only rank 0 writes (safe for distributed training)
- **Auto-flush** - Ensures data is written to disk immediately
- **Error handling** - Continues on errors, logs warnings

**Example:**

```python
from mycelia.shared.metrics import MetricLogger

metric_logger = MetricLogger(config, rank=0)

# Log training metrics
metrics = {
    "step": 100,
    "loss": 2.456,
    "perplexity": 11.65,
    "learning_rate": 0.0001,
    "tokens_per_second": 1024.5,
}
metric_logger.log(metrics, print_log=True)

# Log validation metrics (different keys are OK)
val_metrics = {
    "step": 100,
    "val_loss": 2.123,
    "val_accuracy": 0.945,
}
metric_logger.log(val_metrics, print_log=True)
```

**CSV Output:**

```csv
learning_rate,loss,perplexity,step,tokens_per_second,val_accuracy,val_loss
0.0001,2.456,11.65,100,1024.5,,
,,,100,,0.945,2.123
```

Note: Empty cells for missing keys are handled automatically.

---

### `close()`

Close the metric logger and clean up resources.

```python
def close(self) -> None
```

**Parameters:**
- None

**Returns:**
- `None`

**Side Effects:**
- Closes CSV file
- Finishes W&B run if active
- Logs completion messages

**Example:**

```python
metric_logger = MetricLogger(config, rank=0)

try:
    # ... training loop ...
    for step in range(1000):
        metrics = train_step()
        metric_logger.log(metrics)
finally:
    metric_logger.close()  # Ensure cleanup happens
```

**Automatic Cleanup:**

The logger also implements `__del__()` for automatic cleanup, but explicit `close()` is recommended.

---

## Configuration

### Required Config Structure

```python
class LogConfig:
    log_wandb: bool = False
    wandb_project_name: str = "my-project"
    wandb_resume: bool = False
    wandb_full_id: str = "unique-run-id"  # Optional, for resuming
    metric_path: str = "/path/to/metrics.csv"
```

### YAML Configuration Example

```yaml
log:
  log_wandb: false
  wandb_project_name: "mycelia-training"
  wandb_resume: false
  wandb_full_id: "abc123xyz"
  base_metric_path: "/home/user/metrics"
  metric_path: "/home/user/metrics/run1.csv"
  metric_interval: 20
```

---

## Weights & Biases Integration

### Enabling W&B

Set `log_wandb: true` in your config:

```yaml
log:
  log_wandb: true
  wandb_project_name: "mycelia-subnet"
  wandb_resume: false
```

### W&B Initialization

When enabled, the logger automatically:

1. Initializes W&B with project name
2. Sets run name from `config.run.run_name`
3. Logs full config as W&B config
4. Optionally resumes from run ID

**Example with W&B:**

```python
from mycelia.shared.metrics import MetricLogger

# Config with W&B enabled
config.log.log_wandb = True
config.log.wandb_project_name = "my-training"
config.run.run_name = "experiment-1"

metric_logger = MetricLogger(config, rank=0)

# Metrics are logged to both CSV and W&B
metric_logger.log({"step": 1, "loss": 2.5})
metric_logger.log({"step": 2, "loss": 2.3})

metric_logger.close()  # Calls wandb.finish()
```

---

## Distributed Training

The logger is designed for distributed training:

```python
import torch.distributed as dist

# Each process gets rank
rank = dist.get_rank() if dist.is_initialized() else 0
world_size = dist.get_world_size() if dist.is_initialized() else 1

# Only rank 0 writes metrics
metric_logger = MetricLogger(config, rank=rank)

# All processes can call log(), but only rank 0 writes
for step in range(1000):
    metrics = train_step()
    metric_logger.log(metrics)  # Only rank 0 writes to file/W&B
```

**Benefits:**
- Prevents duplicate writes in multi-GPU training
- All processes can safely call `log()`
- No synchronization overhead

---

## Error Handling

### CSV Write Errors

If CSV writing fails, the logger:
- Logs an error via `structlog`
- Continues execution (non-fatal)
- Includes error details in log

```python
# Error logged but training continues
{"event": "Failed to write metrics to CSV", "error": "...", "path": "/metrics.csv", "level": "error"}
```

### W&B Errors

If W&B initialization or logging fails:
- Logs warning and disables W&B
- Continues with CSV-only logging
- Training is not interrupted

---

## Complete Usage Example

```python
from mycelia.shared.metrics import MetricLogger
from mycelia.shared.config import MinerConfig
import time

# Load config
config = MinerConfig.from_path("config.yaml")

# Initialize logger
metric_logger = MetricLogger(config, rank=0)

try:
    # Training loop
    for step in range(1000):
        start_time = time.time()

        # ... training code ...
        loss = train_step(model, batch)

        # Compute metrics
        step_time = time.time() - start_time
        metrics = {
            "step": step,
            "inner_opt_step": step // 5,
            "loss": loss.item(),
            "perplexity": torch.exp(loss).item(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "step_time_seconds": step_time,
            "tokens_per_second": batch_size * seq_len / step_time,
        }

        # Log every 20 steps
        if step % 20 == 0:
            metric_logger.log(metrics, print_log=True)

        # Validation metrics (different keys)
        if step % 100 == 0:
            val_loss = validate(model, val_loader)
            val_metrics = {
                "step": step,
                "val_loss": val_loss.item(),
                "val_perplexity": torch.exp(val_loss).item(),
            }
            metric_logger.log(val_metrics, print_log=True)

finally:
    # Always close to ensure cleanup
    metric_logger.close()
    print(f"Metrics saved to: {config.log.metric_path}")
```

---

## Best Practices

### DO:

✅ **Use consistent metric names**
```python
# Good - consistent naming
metric_logger.log({"step": 100, "loss": 2.5, "val_loss": 2.3})
```

✅ **Always close the logger**
```python
try:
    metric_logger.log(metrics)
finally:
    metric_logger.close()
```

✅ **Log at regular intervals**
```python
if step % config.log.metric_interval == 0:
    metric_logger.log(metrics)
```

✅ **Include step/timestamp for correlation**
```python
metrics = {"step": step, "epoch": epoch, "loss": loss}
```

### DON'T:

❌ **Don't log every step in tight loops**
```python
# Bad - creates huge files
for step in range(1000000):
    metric_logger.log({"step": step, "loss": loss})

# Good - log periodically
for step in range(1000000):
    if step % 100 == 0:
        metric_logger.log({"step": step, "loss": loss})
```

❌ **Don't log from all ranks**
```python
# Bad - creates duplicate entries
metric_logger = MetricLogger(config, rank=rank)  # rank could be 1, 2, 3...

# Good - only rank 0
metric_logger = MetricLogger(config, rank=0)  # or check dist.get_rank()
```

❌ **Don't mix data types for same metric**
```python
# Bad - inconsistent types
metric_logger.log({"loss": 2.5})      # float
metric_logger.log({"loss": "2.5"})    # string

# Good - consistent types
metric_logger.log({"loss": 2.5})
metric_logger.log({"loss": 2.3})
```

---

## Output Files

### CSV Format

The CSV file uses:
- **Headers** - Column names from metric keys (sorted alphabetically)
- **Dynamic columns** - New keys add new columns automatically
- **Missing values** - Empty cells for metrics not in that row

### Example CSV

```csv
epoch,learning_rate,loss,step,tokens_per_second,val_accuracy,val_loss
1,0.0001,2.456,100,1024.5,,
1,0.0001,2.234,200,1050.2,,
1,0.0001,2.123,300,1032.1,0.945,2.001
2,0.00009,2.089,400,1045.8,,
```

---

## References

- Implementation: `mycelia/shared/metrics.py`
- Used by: `mycelia/miner/train.py`, `mycelia/validator/run.py`
- Related: [Logging API](./app_logging_api.md)
- External: [Weights & Biases Docs](https://docs.wandb.ai/)
