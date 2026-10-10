"""
Runs supervised LoRA/QLoRA fine-tuning using Hugging Face Trainer.

The implementation is designed to remain compatible with different
Hugging Face Transformers TrainingArguments / Trainer APIs.
"""

import inspect
import logging
import os
from typing import Any, Dict

from datasets import DatasetDict
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


logger = logging.getLogger(__name__)


class EpochLossLogger(TrainerCallback):
    """Log training and evaluation loss after each epoch."""

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:

        if not state.log_history:
            return

        # Search backwards because the last log entry may contain
        # only training loss OR evaluation loss.
        train_loss = "N/A"
        eval_loss = "N/A"

        for log in reversed(state.log_history):
            if train_loss == "N/A" and "loss" in log:
                train_loss = log["loss"]

            if eval_loss == "N/A" and "eval_loss" in log:
                eval_loss = log["eval_loss"]

            if train_loss != "N/A" and eval_loss != "N/A":
                break

        epoch = int(state.epoch or 0)

        logger.info(
            "Epoch %d complete — train_loss: %s | eval_loss: %s",
            epoch,
            (
                f"{train_loss:.4f}"
                if isinstance(train_loss, (float, int))
                else train_loss
            ),
            (
                f"{eval_loss:.4f}"
                if isinstance(eval_loss, (float, int))
                else eval_loss
            ),
        )


def _supported_parameters(callable_obj) -> set:
    """
    Return the parameter names supported by the installed Transformers API.
    """

    return set(
        inspect.signature(callable_obj).parameters.keys()
    )


def _add_if_supported(
    kwargs: Dict[str, Any],
    supported: set,
    name: str,
    value: Any,
) -> None:
    """
    Add an argument only if the installed API supports it.
    """

    if name in supported:
        kwargs[name] = value
    else:
        logger.info(
            "Transformers API does not support '%s' — skipping.",
            name,
        )


def build_training_arguments(
    training_cfg: Dict[str, Any],
) -> TrainingArguments:
    """
    Build TrainingArguments while automatically adapting to the
    installed Transformers version.
    """

    supported = _supported_parameters(
        TrainingArguments.__init__
    )

    kwargs: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core training arguments
    # ------------------------------------------------------------------

    _add_if_supported(
        kwargs,
        supported,
        "output_dir",
        training_cfg["output_dir"],
    )

    _add_if_supported(
        kwargs,
        supported,
        "num_train_epochs",
        training_cfg["num_train_epochs"],
    )

    _add_if_supported(
        kwargs,
        supported,
        "per_device_train_batch_size",
        training_cfg["per_device_train_batch_size"],
    )

    _add_if_supported(
        kwargs,
        supported,
        "per_device_eval_batch_size",
        training_cfg["per_device_eval_batch_size"],
    )

    _add_if_supported(
        kwargs,
        supported,
        "gradient_accumulation_steps",
        training_cfg["gradient_accumulation_steps"],
    )

    _add_if_supported(
        kwargs,
        supported,
        "learning_rate",
        training_cfg["learning_rate"],
    )

    _add_if_supported(
        kwargs,
        supported,
        "lr_scheduler_type",
        training_cfg["lr_scheduler_type"],
    )

    _add_if_supported(
        kwargs,
        supported,
        "warmup_steps",
        training_cfg.get("warmup_steps", 0),
    )

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------

    _add_if_supported(
        kwargs,
        supported,
        "fp16",
        training_cfg.get("fp16", False),
    )

    _add_if_supported(
        kwargs,
        supported,
        "bf16",
        training_cfg.get("bf16", False),
    )

    # ------------------------------------------------------------------
    # Evaluation strategy
    #
    # Newer Transformers:
    #     eval_strategy
    #
    # Older Transformers:
    #     evaluation_strategy
    # ------------------------------------------------------------------

    eval_strategy = training_cfg.get(
        "eval_strategy",
        "epoch",
    )

    if "eval_strategy" in supported:

        kwargs["eval_strategy"] = eval_strategy

    elif "evaluation_strategy" in supported:

        kwargs["evaluation_strategy"] = eval_strategy

    else:

        logger.warning(
            "Installed Transformers API supports neither "
            "'eval_strategy' nor 'evaluation_strategy'. "
            "Evaluation strategy will not be explicitly configured."
        )

    # ------------------------------------------------------------------
    # Saving strategy
    # ------------------------------------------------------------------

    save_strategy = training_cfg.get(
        "save_strategy",
        "epoch",
    )

    _add_if_supported(
        kwargs,
        supported,
        "save_strategy",
        save_strategy,
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    _add_if_supported(
        kwargs,
        supported,
        "logging_steps",
        training_cfg.get("logging_steps", 10),
    )

    _add_if_supported(
        kwargs,
        supported,
        "report_to",
        training_cfg.get("report_to", "none"),
    )

    # ------------------------------------------------------------------
    # Best model handling
    #
    # This is only enabled when evaluation and saving strategies match.
    # ------------------------------------------------------------------

    strategies_match = (
        eval_strategy == save_strategy
    )

    load_best_model = (
        training_cfg.get(
            "load_best_model_at_end",
            True,
        )
        and strategies_match
    )

    _add_if_supported(
        kwargs,
        supported,
        "load_best_model_at_end",
        load_best_model,
    )

    if load_best_model:

        _add_if_supported(
            kwargs,
            supported,
            "metric_for_best_model",
            "eval_loss",
        )

        _add_if_supported(
            kwargs,
            supported,
            "greater_is_better",
            False,
        )

    # ------------------------------------------------------------------
    # Checkpoint limit
    # ------------------------------------------------------------------

    _add_if_supported(
        kwargs,
        supported,
        "save_total_limit",
        training_cfg.get(
            "save_total_limit",
            1,
        ),
    )

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------

    _add_if_supported(
        kwargs,
        supported,
        "dataloader_pin_memory",
        training_cfg.get(
            "dataloader_pin_memory",
            True,
        ),
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    _add_if_supported(
        kwargs,
        supported,
        "optim",
        training_cfg.get(
            "optim",
            "adamw_torch",
        ),
    )

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    _add_if_supported(
        kwargs,
        supported,
        "gradient_checkpointing",
        training_cfg.get(
            "gradient_checkpointing",
            False,
        ),
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT pass group_by_length.
    #
    # The installed Transformers version on the test machine does not
    # support it.
    # ------------------------------------------------------------------

    logger.info(
        "Building TrainingArguments using supported parameters."
    )

    logger.info(
        "Supported TrainingArguments parameters detected: %d",
        len(supported),
    )

    return TrainingArguments(**kwargs)


def run_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tokenized_dataset: DatasetDict,
    training_cfg: Dict[str, Any],
) -> Trainer:
    """
    Execute supervised fine-tuning.

    Returns the trained Trainer instance so the pipeline can retrieve
    training history and the trained PEFT model.
    """

    if (
        "train" not in tokenized_dataset
        or "val" not in tokenized_dataset
    ):
        raise KeyError(
            "tokenized_dataset must contain "
            "'train' and 'val' splits."
        )

    os.makedirs(
        training_cfg["output_dir"],
        exist_ok=True,
    )

    # Ensure a padding token exists.
    if tokenizer.pad_token is None:

        if tokenizer.eos_token is not None:

            tokenizer.pad_token = tokenizer.eos_token

            logger.info(
                "pad_token was missing — using eos_token: %s",
                tokenizer.eos_token,
            )

        else:

            raise ValueError(
                "Tokenizer has neither pad_token nor eos_token."
            )

    training_args = build_training_arguments(
        training_cfg
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # ------------------------------------------------------------------
    # Trainer API compatibility
    #
    # Newer Transformers uses:
    #     processing_class
    #
    # Older Transformers uses:
    #     tokenizer
    # ------------------------------------------------------------------

    trainer_supported = _supported_parameters(
        Trainer.__init__
    )

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["val"],
        "data_collator": data_collator,
        "callbacks": [
            EpochLossLogger()
        ],
    }

    if "processing_class" in trainer_supported:

        trainer_kwargs["processing_class"] = tokenizer

        logger.info(
            "Using Trainer API: processing_class"
        )

    elif "tokenizer" in trainer_supported:

        trainer_kwargs["tokenizer"] = tokenizer

        logger.info(
            "Using Trainer API: tokenizer"
        )

    else:

        logger.warning(
            "Installed Trainer API supports neither "
            "'processing_class' nor 'tokenizer'."
        )

    trainer = Trainer(
        **trainer_kwargs
    )

    logger.info(
        "Starting fine-tuning..."
    )

    logger.info(
        "Train examples: %d | Val examples: %d | Epochs: %d",
        len(tokenized_dataset["train"]),
        len(tokenized_dataset["val"]),
        training_cfg["num_train_epochs"],
    )

    trainer.train()

    logger.info(
        "Fine-tuning complete."
    )

    return trainer