from typing import Any
from functools import partial

from datasets import interleave_datasets, load_dataset
from transformers import PreTrainedTokenizerBase

from mycelia.shared.dataloader import DefaultStreamingTorchDataset


# -------------------------------------------------------------
# Customer Extension Point: Customize how your dataset is loaded
# make sure this class was pointed to in the config through config.task.exp.data.dataset_class
# -------------------------------------------------------------
class StreamingTorchDataset(DefaultStreamingTorchDataset):
    @staticmethod
    def tokenize_and_format(
        example: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
    ) -> dict[str, Any]:
        """
        Default data formatting function.

        This function demonstrates how to transform a raw dataset row
        into tokenized tensors suitable for training. Customers may
        freely modify or replace this function to implement custom
        formatting logic.

        Expected Input
        --------------
        example : Dict[str, Any]
            A single raw dataset row containing a list of chat-style messages
            under `example["messages"]`.

        tokenizer : PreTrainedTokenizerBase
            HuggingFace tokenizer used to build and tokenize the text sequence.

        sequence_length : int
            Maximum sequence length for tokenization and padding.

        Expected Output
        ---------------
        Dict[str, Any]
            Dictionary containing tokenized tensors:

                {
                    "input_ids": Tensor[1, sequence_length],
                    "attention_mask": Tensor[1, sequence_length]
                }

            Additional fields may be added if required by your model.

        Notes for Customization
        -----------------------
        - You may change how messages are converted to text.
        - You can modify tokenization parameters (padding, truncation, etc.).
        - You can inject additional metadata into the output dictionary.
        - You can apply your own chat template logic.

        Returns
        -------
        Dict[str, Any]
            Tokenized output ready to be consumed by a DataLoader.
        """
        # 1) Convert dataset row → chat messages or text
        if "messages" in example:
            # 2) Convert messages → raw text using model's chat template
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif "text" in example:
            # Fallback for datasets with "text" field (e.g., C4)
            text = example["text"]
        else:
            # Last resort: concatenate string fields
            text = " ".join(str(v) for v in example.values() if isinstance(v, str))

        # 3) Tokenize text → tensors
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


# Merged math dataset that loads multiple math datasets
class MergedMathDataset(StreamingTorchDataset):
    """
    Loads and merges multiple math datasets from HuggingFace Hub.

    Datasets included:
    - openai/gsm8k: Grade school math problems
    - lighteval/MATH: Competition mathematics
    - meta-math/MetaMathQA: Augmented math reasoning
    """

    # Define the math datasets to merge
    MATH_DATASETS = [
        ("openai/gsm8k", "main", 0.5),  # (name, config/split, weight)
        ("meta-math/MetaMathQA", None, 0.5),  # Augmented math reasoning
        # Note: Add more datasets as needed. Examples:
        # ("EleutherAI/math_qa", None, 0.2),
        # ("ChilleD/SVAMP", None, 0.1),
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
                logger.info(f"Loading dataset", name=dataset_name, config=config_name)
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
                logger.info(f"Loaded dataset", name=dataset_name, split=split_name)
            except Exception as e:
                logger.error(f"Failed to load dataset", name=dataset_name, error=str(e))
                continue

        if not datasets:
            raise ValueError("No datasets could be loaded")

        # Normalize probabilities
        total_weight = sum(probabilities)
        probabilities = [p / total_weight for p in probabilities]

        logger.info(f"Merging {len(datasets)} datasets with probabilities", probs=probabilities)

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
