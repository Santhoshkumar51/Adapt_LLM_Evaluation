"""
Saves the trained LoRA adapter separately from the base model.

The output directory contains the adapter weights and PEFT configuration,
not a merged standalone copy of the base model.
"""

import logging
import os

from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def merge_and_save(
    base_model: PreTrainedModel,
    finetuned_model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    adapter_output_dir: str,
) -> PeftModel:
    """
    Save the trained PEFT adapter without merging it into
    the base model.

    base_model is retained in the function signature for
    compatibility with the existing pipeline.
    """

    if (
        not adapter_output_dir
        or not adapter_output_dir.strip()
    ):
        raise ValueError(
            "adapter_output_dir must be "
            "a non-empty path string."
        )

    os.makedirs(
        adapter_output_dir,
        exist_ok=True
    )

    logger.info(
        "Saving LoRA adapter to: %s",
        adapter_output_dir
    )

    finetuned_model.save_pretrained(
        adapter_output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        adapter_output_dir
    )

    for fname in os.listdir(
        adapter_output_dir
    ):

        fpath = os.path.join(
            adapter_output_dir,
            fname
        )

        if os.path.isfile(fpath):

            size_mb = (
                os.path.getsize(fpath)
                / (1024 * 1024)
            )

            logger.info(
                "Saved: %s (%.2f MB)",
                fname,
                size_mb
            )

    logger.info(
        "LoRA adapter saved separately "
        "from the base model."
    )

    return finetuned_model