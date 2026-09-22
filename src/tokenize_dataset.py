"""
Loads processed instruction-response .jsonl files and tokenizes them
into model-ready input tensors. Handles padding, truncation, and
label masking so the model only learns to predict the response tokens,
not the instruction tokens.
"""

import logging
from typing import Dict

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Instruction delimiter — model learns to predict tokens after this only
RESPONSE_DELIMITER = "### Response:"


def _tokenize_and_mask(
    examples: Dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict:
    """
    Tokenizes a batch of examples and masks instruction tokens in labels.
    The model is trained to predict response tokens only — instruction
    tokens are masked with -100 so they are ignored by the loss function.

    Args:
        examples:   Batch of examples with a 'text' field.
        tokenizer:  Tokenizer matching the base model.
        max_length: Maximum token sequence length.

    Returns:
        Dict with input_ids, attention_mask, and labels.
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,    # return lists, not tensors — datasets handles batching
    )

    labels = []
    for i, input_ids in enumerate(tokenized["input_ids"]):
        # Decode to find where the response starts
        full_text = examples["text"][i]
        delimiter_pos = full_text.find(RESPONSE_DELIMITER)

        if delimiter_pos == -1:
            # No delimiter found — train on full sequence (fallback)
            logger.warning("Response delimiter not found in example %d. Training on full sequence.", i)
            labels.append(input_ids.copy())
            continue

        # Tokenize only the instruction portion to find its token length
        instruction_part = full_text[:delimiter_pos + len(RESPONSE_DELIMITER)]
        instruction_tokens = tokenizer(
            instruction_part,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        instruction_length = len(instruction_tokens["input_ids"])

        # Mask instruction tokens with -100 — ignored by CrossEntropyLoss
        masked_labels = [-100] * instruction_length + input_ids[instruction_length:]
        # Pad labels to max_length with -100
        masked_labels = masked_labels[:max_length]
        masked_labels += [-100] * (max_length - len(masked_labels))
        labels.append(masked_labels)

    tokenized["labels"] = labels
    return tokenized


def tokenize_dataset(
    train_file: str,
    val_file: str,
    test_file: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> DatasetDict:
    """
    Loads train/val/test .jsonl splits and returns a tokenized DatasetDict.

    Args:
        train_file:  Path to train.jsonl.
        val_file:    Path to val.jsonl.
        test_file:   Path to test.jsonl.
        tokenizer:   Tokenizer matching the base model.
        max_length:  Maximum sequence length for truncation and padding.

    Returns:
        DatasetDict with 'train', 'val', and 'test' splits — tokenized and masked.

    Raises:
        FileNotFoundError: If any of the split files do not exist.
    """
    logger.info("Loading dataset splits...")

    raw_dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
            "test": test_file,
        },
    )

    logger.info(
        "Split sizes — Train: %d | Val: %d | Test: %d",
        len(raw_dataset["train"]),
        len(raw_dataset["val"]),
        len(raw_dataset["test"]),
    )

    logger.info("Tokenizing dataset with max_length=%d ...", max_length)

    tokenized_dataset = raw_dataset.map(
        lambda examples: _tokenize_and_mask(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],   # drop raw text — model only needs token IDs
        desc="Tokenizing",
    )

    logger.info("Tokenization complete.")
    return tokenized_dataset