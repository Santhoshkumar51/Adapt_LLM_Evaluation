"""
Loads a pretrained causal LM and its tokenizer from Hugging Face Hub.
Supports 4-bit quantization (QLoRA) via bitsandbytes for memory-efficient
loading on single consumer-grade GPUs.
"""

import logging
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def build_bnb_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    """
    Constructs a BitsAndBytesConfig for 4-bit quantization (QLoRA).
    Returns None if quantization is not requested.

    Args:
        load_in_4bit: Whether to load the model in 4-bit precision.

    Returns:
        BitsAndBytesConfig if 4-bit, else None.
    """
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,   # nested quantization for extra memory saving
        bnb_4bit_quant_type="nf4",        # NormalFloat4 — optimal for normally distributed weights
    )


def load_base_model(
    model_name: str,
    load_in_4bit: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads a causal language model and tokenizer from Hugging Face Hub.
    The model is loaded with optional 4-bit quantization for QLoRA training.
    Model weights are frozen by default — LoRA adapters are added separately.

    Args:
        model_name:    Hugging Face model ID, e.g. 'mistralai/Mistral-7B-v0.1'.
        load_in_4bit:  If True, loads model in 4-bit precision via bitsandbytes.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ValueError: If model_name is empty or None.
        RuntimeError: If model loading fails due to CUDA/memory issues.
    """
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty Hugging Face model ID.")

    logger.info("Loading tokenizer for: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    # Ensure pad token exists — many causal LMs omit it by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token not found — set to eos_token: '%s'", tokenizer.eos_token)

    bnb_config = build_bnb_config(load_in_4bit)

    logger.info(
        "Loading model: %s | 4-bit quantization: %s",
        model_name,
        load_in_4bit,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",             # automatically distributes across available GPUs/CPU
        trust_remote_code=True,
        torch_dtype=torch.float16 if not load_in_4bit else None,
    )

    # Disable caching — incompatible with gradient checkpointing during training
    model.config.use_cache = False
    model.config.pretraining_tp = 1   # tensor parallelism — set to 1 for single GPU

    logger.info("Model loaded successfully. Parameters: %s", f"{model.num_parameters():,}")

    return model, tokenizer