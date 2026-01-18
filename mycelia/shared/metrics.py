"""Metrics logging for the Mycelia project."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MetricLogger:
    """Handles logging of training/validation metrics to CSV and optionally to Weights & Biases."""

    def __init__(self, config: Any, rank: int = 0) -> None:
        """Initialize the metric logger.

        Args:
            config: The configuration object containing logging settings
            rank: The process rank (for distributed training)
        """
        self.config = config
        self.rank = rank
        self.metric_path = Path(config.log.metric_path)
        self.log_wandb = config.log.log_wandb
        self.wandb_run = None

        # Create metrics directory if it doesn't exist
        self.metric_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file with headers
        self.csv_file = None
        self.csv_writer = None
        self.headers_written = False

        # Initialize wandb if enabled
        if self.log_wandb and rank == 0:
            try:
                import wandb

                wandb_kwargs = {
                    "project": config.log.wandb_project_name,
                    "name": config.run.run_name,
                    "config": self._config_to_dict(config),
                    "resume": config.log.wandb_resume,
                }

                # Add wandb run ID if resuming
                if config.log.wandb_resume and hasattr(config.log, "wandb_full_id"):
                    wandb_kwargs["id"] = config.log.wandb_full_id

                self.wandb_run = wandb.init(**wandb_kwargs)
                logger.info("Initialized wandb logging", project=config.log.wandb_project_name)
            except ImportError:
                logger.warning("wandb is not installed, skipping wandb logging")
                self.log_wandb = False
            except Exception as e:
                logger.error("Failed to initialize wandb", error=str(e))
                self.log_wandb = False

    def _config_to_dict(self, config: Any) -> dict[str, Any]:
        """Convert config object to dictionary for wandb."""
        if hasattr(config, "model_dump"):
            return config.model_dump()
        elif hasattr(config, "dict"):
            return config.dict()
        else:
            # Fallback: try to convert to dict manually
            result = {}
            for key in dir(config):
                if not key.startswith("_"):
                    try:
                        value = getattr(config, key)
                        if not callable(value):
                            result[key] = value
                    except Exception:
                        pass
            return result

    def log(self, metrics: dict[str, Any], print_log: bool = True) -> None:
        """Log metrics to CSV and optionally to wandb.

        Args:
            metrics: Dictionary of metric names and values
            print_log: Whether to print the metrics to the logger
        """
        # Only log from rank 0 to avoid duplicate writes
        if self.rank != 0:
            return

        # Initialize CSV file if not already done
        if self.csv_file is None:
            self.csv_file = open(self.metric_path, "a", newline="")
            # Start with empty fieldnames - will be updated dynamically
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[], extrasaction='ignore')
            self.all_fieldnames = set()

        # Update fieldnames if new fields are present
        current_fields = set(metrics.keys())
        if current_fields - self.all_fieldnames:
            # New fields detected - need to reinitialize writer
            self.all_fieldnames.update(current_fields)

            # Close and reopen file to update headers if needed
            if not self.headers_written and self.metric_path.stat().st_size == 0:
                # First write - write headers
                self.csv_writer = csv.DictWriter(
                    self.csv_file,
                    fieldnames=sorted(self.all_fieldnames),
                    extrasaction='ignore'
                )
                self.csv_writer.writeheader()
                self.headers_written = True
            else:
                # Update writer with new fieldnames
                self.csv_writer = csv.DictWriter(
                    self.csv_file,
                    fieldnames=sorted(self.all_fieldnames),
                    extrasaction='ignore'
                )

        # Write metrics to CSV
        try:
            self.csv_writer.writerow(metrics)
            self.csv_file.flush()
        except Exception as e:
            logger.error("Failed to write metrics to CSV", error=str(e), path=str(self.metric_path))

        # Log to wandb if enabled
        if self.log_wandb and self.wandb_run is not None:
            try:
                import wandb

                wandb.log(metrics)
            except Exception as e:
                logger.error("Failed to log to wandb", error=str(e))

        # Print metrics if requested
        if print_log:
            logger.info("Metrics", **metrics)

    def close(self) -> None:
        """Close the metric logger and clean up resources."""
        if self.rank != 0:
            return

        # Close CSV file
        if self.csv_file is not None:
            self.csv_file.close()
            logger.info("Closed metric CSV file", path=str(self.metric_path))

        # Finish wandb run
        if self.log_wandb and self.wandb_run is not None:
            try:
                import wandb

                wandb.finish()
                logger.info("Finished wandb run")
            except Exception as e:
                logger.error("Failed to finish wandb run", error=str(e))

    def __del__(self) -> None:
        """Ensure resources are cleaned up when object is garbage collected."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
