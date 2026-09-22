"""
Merges trained LoRA adapter weights into the base model's frozen weights,
producing a single standalone model that requires no PEFT library at inference.
Saves the merged model and tokenizer to the adapter output directory.
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
) -> PreTrainedModel:
    """
    Mathematically merges LoRA adapter weights (BA matrices) into the base
    model's frozen weights (W → W + BA), then saves the merged model and
    tokenizer to disk. The resulting model is a standard HF model with no
    PEFT dependency — ready for standalone serving.

    Args:
        base_model:         Original frozen base model (pre-adapter).
        finetuned_model:    PEFT model with trained LoRA adapters.
        tokenizer:          Tokenizer to save alongside the merged model.
        adapter_output_dir: Directory to save merged weights and config.

    Returns:
        The merged PreTrainedModel with adapter weights baked in.

    Raises:
        ValueError: If adapter_output_dir is empty or None.
        RuntimeError: If merging fails due to dtype or device mismatch.
    """
    if not adapter_output_dir or not adapter_output_dir.strip():
        raise ValueError("adapter_output_dir must be a non-empty path string.")

    os.makedirs(adapter_output_dir, exist_ok=True)

    logger.info("Merging LoRA adapter weights into base model...")

    # merge_and_unload() adds BA to W for each adapted layer,
    # then detaches the adapter — returns a plain PreTrainedModel
    merged_model = finetuned_model.merge_and_unload()

    logger.info("Merge complete. Saving merged model to: %s", adapter_output_dir)

    merged_model.save_pretrained(
        adapter_output_dir,
        safe_serialization=True,    # save as .safetensors, not .bin
    )

    tokenizer.save_pretrained(adapter_output_dir)

    # Log final adapter file size for reporting in the dashboard
    for fname in os.listdir(adapter_output_dir):
        fpath = os.path.join(adapter_output_dir, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            logger.info("Saved: %s (%.2f MB)", fname, size_mb)

    logger.info("Adapter saved successfully.")
    return merged_model