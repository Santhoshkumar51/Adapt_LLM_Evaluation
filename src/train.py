"""
Runs supervised LoRA/QLoRA fine-tuning using Hugging Face Trainer.
"""

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

        last_log = state.log_history[-1]

        train_loss = last_log.get(
            "loss",
            "N/A",
        )

        eval_loss = last_log.get(
            "eval_loss",
            "N/A",
        )

        epoch = int(
            state.epoch or 0
        )

        logger.info(
            "Epoch %d complete — "
            "train_loss: %s | eval_loss: %s",
            epoch,
            (
                f"{train_loss:.4f}"
                if isinstance(
                    train_loss,
                    float,
                )
                else train_loss
            ),
            (
                f"{eval_loss:.4f}"
                if isinstance(
                    eval_loss,
                    float,
                )
                else eval_loss
            ),
        )


def build_training_arguments(
    training_cfg: Dict[str, Any],
) -> TrainingArguments:
    """Build TrainingArguments from the YAML configuration."""

    return TrainingArguments(
        output_dir=training_cfg[
            "output_dir"
        ],

        num_train_epochs=training_cfg[
            "num_train_epochs"
        ],

        per_device_train_batch_size=training_cfg[
            "per_device_train_batch_size"
        ],

        per_device_eval_batch_size=training_cfg[
            "per_device_eval_batch_size"
        ],

        gradient_accumulation_steps=training_cfg[
            "gradient_accumulation_steps"
        ],

        learning_rate=training_cfg[
            "learning_rate"
        ],

        lr_scheduler_type=training_cfg[
            "lr_scheduler_type"
        ],

        warmup_steps=training_cfg.get(
            "warmup_steps",
            0,
        ),

        fp16=training_cfg.get(
            "fp16",
            False,
        ),

        evaluation_strategy=training_cfg[
            "eval_strategy"
        ],

        save_strategy=training_cfg[
            "save_strategy"
        ],

        logging_steps=training_cfg[
            "logging_steps"
        ],

        report_to=training_cfg.get(
            "report_to",
            "none",
        ),

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        save_total_limit=training_cfg.get(
            "save_total_limit",
            1,
        ),

        dataloader_pin_memory=True,

        group_by_length=True,

        optim=training_cfg.get(
            "optim",
            "adamw_torch",
        ),

        gradient_checkpointing=training_cfg.get(
            "gradient_checkpointing",
            False,
        ),
    )


def run_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tokenized_dataset: DatasetDict,
    training_cfg: Dict[str, Any],
) -> Trainer:
    """
    Execute supervised fine-tuning.

    Returns the trained Trainer instance so the pipeline can
    retrieve training history and the trained PEFT model.
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

    training_args = build_training_arguments(
        training_cfg
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset[
            "train"
        ],
        eval_dataset=tokenized_dataset[
            "val"
        ],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EpochLossLogger()
        ],
    )

    logger.info(
        "Starting fine-tuning..."
    )

    logger.info(
        "Train examples: %d | "
        "Val examples: %d | "
        "Epochs: %d",
        len(tokenized_dataset["train"]),
        len(tokenized_dataset["val"]),
        training_cfg["num_train_epochs"],
    )

    trainer.train()

    logger.info(
        "Fine-tuning complete."
    )

    return trainer