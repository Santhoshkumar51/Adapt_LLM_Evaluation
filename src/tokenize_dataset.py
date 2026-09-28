"""
Loads processed instruction-response JSONL files and tokenizes them.

The original `text` field is preserved because the evaluation stage
uses the formatted instruction-response strings to generate prompts,
extract references, and compute perplexity.
"""

import logging
from typing import Dict

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)

RESPONSE_DELIMITER = "### Response:"


def _tokenize_and_mask(
    examples: Dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict:
    """
    Tokenize a batch of formatted examples and mask instruction
    tokens in the training labels.

    The raw `text` field is intentionally NOT removed.
    """

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    labels = []

    for i, input_ids in enumerate(
        tokenized["input_ids"]
    ):
        full_text = examples["text"][i]

        delimiter_pos = full_text.find(
            RESPONSE_DELIMITER
        )

        if delimiter_pos == -1:
            logger.warning(
                "Response delimiter not found "
                "in example %d. Training on the "
                "complete sequence.",
                i,
            )

            labels.append(input_ids.copy())
            continue

        instruction_part = (
            full_text[
                :delimiter_pos
                + len(RESPONSE_DELIMITER)
            ]
        )

        instruction_tokens = tokenizer(
            instruction_part,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

        instruction_length = len(
            instruction_tokens["input_ids"]
        )

        masked_labels = (
            [-100] * instruction_length
            + input_ids[instruction_length:]
        )

        masked_labels = masked_labels[
            :max_length
        ]

        masked_labels += (
            [-100]
            * (
                max_length
                - len(masked_labels)
            )
        )

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
    Load train/validation/test JSONL files,
    tokenize them, and preserve the original
    formatted text column for evaluation.
    """

    logger.info(
        "Loading dataset splits..."
    )

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

    logger.info(
        "Tokenizing with max_length=%d...",
        max_length,
    )

    tokenized_dataset = raw_dataset.map(
        lambda examples:
            _tokenize_and_mask(
                examples,
                tokenizer,
                max_length,
            ),
        batched=True,
        desc="Tokenizing",
    )

    logger.info(
        "Tokenization complete."
    )

    return tokenized_dataset