"""
Injects LoRA adapter layers into a loaded base model using the PEFT library.
Only the adapter parameters are set as trainable — all base model weights
remain frozen throughout training.
"""

import logging
from typing import Any, Dict

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def _log_trainable_parameters(model: PreTrainedModel) -> None:
    """
    Logs the number and percentage of trainable parameters after adapter injection.
    This is the core efficiency metric of LoRA — typically under 1% of total params.

    Args:
        model: PEFT-wrapped model with LoRA adapters injected.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100.0 * trainable / total

    logger.info(
        "Trainable parameters: %s / %s (%.4f%%)",
        f"{trainable:,}",
        f"{total:,}",
        percentage,
    )


def inject_lora_adapters(
    model: PreTrainedModel,
    lora_cfg: Dict[str, Any],
    load_in_4bit: bool = True,
) -> PreTrainedModel:
    """
    Prepares the base model for k-bit training and injects LoRA adapters
    into the specified target modules. Returns the PEFT-wrapped model with
    only adapter parameters marked as trainable.

    Args:
        model:        Loaded base model (frozen, optionally quantized).
        lora_cfg:     Dictionary of LoRA hyperparameters from lora_config.yaml.
                      Expected keys: r, lora_alpha, target_modules,
                      lora_dropout, bias, task_type.
        load_in_4bit: Whether the model was loaded in 4-bit (QLoRA) — determines
                      whether prepare_model_for_kbit_training is applied.

    Returns:
        PEFT model with LoRA adapters injected and only adapter params trainable.

    Raises:
        KeyError: If required LoRA config keys are missing.
        ValueError: If target_modules is empty or not a list.
    """
    required_keys = {"r", "lora_alpha", "target_modules", "lora_dropout", "bias", "task_type"}
    missing = required_keys - set(lora_cfg.keys())
    if missing:
        raise KeyError(f"Missing required LoRA config keys: {missing}")

    if not lora_cfg["target_modules"]:
        raise ValueError("target_modules must be a non-empty list of module names.")

    # Required step before adapter injection for quantized (4-bit/8-bit) models
    # Casts layer norms and output embedding to float32 for stable training
    if load_in_4bit:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,   # trades compute for memory — essential for large models
        )

    task_type = TaskType[lora_cfg["task_type"]]   # e.g. TaskType.CAUSAL_LM

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=task_type,
        inference_mode=False,   # must be False during training
    )

    logger.info(
        "Injecting LoRA adapters | rank=%d | alpha=%d | target_modules=%s",
        lora_cfg["r"],
        lora_cfg["lora_alpha"],
        lora_cfg["target_modules"],
    )

    model = get_peft_model(model, lora_config)

    _log_trainable_parameters(model)

    return model