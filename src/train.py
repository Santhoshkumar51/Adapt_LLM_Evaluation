"""
Runs the supervised fine-tuning loop using Hugging Face Trainer.
Accepts the PEFT model, tokenized dataset, and training config.
Saves checkpoints per epoch and logs training/eval metrics to TensorBoard.
"""

import logging
import os
from typing import Any, Dict

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
from datasets import DatasetDict

logger = logging.getLogger(__name__)


class EpochLossLogger(TrainerCallback):
    """
    Custom callback that logs training and evaluation loss at the
    end of each epoch to both the console and TensorBoard.
    """

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.log_history:
            last_log = state.log_history[-1]
            train_loss = last_log.get("loss", "N/A")
            eval_loss = last_log.get("eval_loss", "N/A")
            logger.info(
                "Epoch %d complete — train_loss: %s | eval_loss: %s",
                int(state.epoch),
                f"{train_loss:.4f}" if isinstance(train_loss, float) else train_loss,
                f"{eval_loss:.4f}" if isinstance(eval_loss, float) else eval_loss,
            )


def build_training_arguments(
    training_cfg: Dict[str, Any],
) -> TrainingArguments:
    """
    Constructs a TrainingArguments object from the training config dictionary.

    Args:
        training_cfg: Training hyperparameters from training_config.yaml.

    Returns:
        TrainingArguments object ready for Trainer.
    """
    return TrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],
        fp16=training_cfg["fp16"],
        evaluation_strategy=training_cfg["eval_strategy"],
        save_strategy=training_cfg["save_strategy"],
        logging_steps=training_cfg["logging_steps"],
        report_to="tensorboard",           # log metrics to TensorBoard
        load_best_model_at_end=True,       # restore best checkpoint after training
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,                # keep only last 2 checkpoints to save disk space
        dataloader_pin_memory=True,
        group_by_length=True,              # batch similar-length sequences — reduces padding waste
    )


def run_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tokenized_dataset: DatasetDict,
    training_cfg: Dict[str, Any],
) -> Trainer:
    """
    Executes the supervised fine-tuning loop using Hugging Face Trainer.
    Logs metrics per epoch via TensorBoard and saves checkpoints to disk.

    Args:
        model:              PEFT model with LoRA adapters injected.
        tokenizer:          Tokenizer matching the base model.
        tokenized_dataset:  DatasetDict with 'train' and 'val' splits.
        training_cfg:       Training hyperparameters from training_config.yaml.

    Returns:
        Trained Trainer instance (use to access training history and model state).

    Raises:
        KeyError: If 'train' or 'val' splits are missing from tokenized_dataset.
    """
    if "train" not in tokenized_dataset or "val" not in tokenized_dataset:
        raise KeyError("tokenized_dataset must contain 'train' and 'val' splits.")

    os.makedirs(training_cfg["output_dir"], exist_ok=True)

    training_args = build_training_arguments(training_cfg)

    # DataCollator handles dynamic padding within each batch at runtime
    # — more efficient than padding everything to max_length statically
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,   # consistent with label masking in tokenize_dataset.py
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EpochLossLogger()],
    )

    logger.info("Starting fine-tuning...")
    logger.info(
        "Train examples: %d | Val examples: %d | Epochs: %d",
        len(tokenized_dataset["train"]),
        len(tokenized_dataset["val"]),
        training_cfg["num_train_epochs"],
    )

    trainer.train()

    logger.info("Fine-tuning complete. Best model restored from checkpoint.")
    return trainer