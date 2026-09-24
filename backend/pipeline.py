"""
AdaptEval pipeline.py

The single glue module that connects FastAPI routes to the src/ training
modules.

TEST MODE:
    Used during local development to verify:
        upload → dataset preparation → job status → frontend polling

    It does NOT load or train a 7B model.

REAL MODE:
    When TEST_MODE = False, the original QLoRA fine-tuning pipeline runs.
"""

import json
import logging
import os
import time
from typing import Any, Dict

from src.load_base_model import load_base_model
from src.tokenize_dataset import tokenize_dataset
from src.inject_lora import inject_lora_adapters
from src.train import run_training
from src.evaluate import run_evaluation
from src.merge_adapter import merge_and_save

import yaml

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG_LORA = "configs/lora_config.yaml"
CONFIG_TRAINING = "configs/training_config.yaml"
ADAPTER_BASE = "adapters"

# ── LOCAL TEST MODE ───────────────────────────────────────────────────────────
#
# True  → Safe local testing. No 7B model is loaded.
# False → Real QLoRA pipeline.
#
# Keep this TRUE while testing on your Intel Iris Xe laptop.
#
TEST_MODE = True


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_status(
    job_store: Dict,
    job_id: str,
    status: str,
    **kwargs
):
    """
    Update job status in the shared JOB_STORE dictionary.

    kwargs can include:
        progress
        results
        error
    """

    job_store[job_id]["status"] = status

    for key, val in kwargs.items():
        job_store[job_id][key] = val

    logger.info(
        "Job %s → %s",
        job_id[:8],
        status
    )


# =============================================================================
# LOCAL TEST PIPELINE
# =============================================================================

def run_test_pipeline(
    job_id: str,
    model_id: str,
    data_file_path: str,
    job_store: Dict[str, Any],
) -> None:
    """
    Safe local development pipeline.

    This verifies:

        File upload
            ↓
        Dataset preparation
            ↓
        Job status updates
            ↓
        Frontend polling
            ↓
        Completion

    IMPORTANT:
        No Hugging Face model is loaded.
        No CUDA is required.
        No LoRA/QLoRA training occurs.
    """

    try:

        # ── Stage 1: PREPARING ────────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "preparing"
        )

        logger.info(
            "TEST MODE: Preparing dataset for job %s",
            job_id[:8]
        )

        # Import here so the test pipeline can directly control
        # the uploaded dataset path.
        import data.prepare_dataset as prep_module

        prep_module.RAW_FILE = data_file_path

        prep_module.OUTPUT_DIR = (
            f"data/processed/{job_id}/"
        )

        # Run the real dataset preparation code.
        prep_module.main()

        logger.info(
            "TEST MODE: Dataset preparation complete."
        )

        # ── Stage 2: TRAINING ────────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "training",
            progress={
                "current_epoch": 1,
                "total_epochs": 1,
                "train_loss": 0.0,
                "eval_loss": 0.0,
                "elapsed_mins": 0.1,
            },
        )

        logger.info(
            "TEST MODE: Skipping actual model training."
        )

        # Simulate a short training period.
        time.sleep(3)

        # ── Stage 3: EVALUATING ──────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "evaluating"
        )

        logger.info(
            "TEST MODE: Skipping model evaluation."
        )

        # Simulate evaluation.
        time.sleep(3)

        # ── Stage 4: COMPLETE ────────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "complete",
            results={
                "baseline": {
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "perplexity": 0.0,
                },

                "finetuned": {
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "perplexity": 0.0,
                },

                "improvement": {
                    "accuracy_delta": 0.0,
                    "f1_delta": 0.0,
                    "perplexity_delta": 0.0,
                },

                "samples": [],

                "adapter": {
                    "size_mb": 0.0,
                    "trained_params_pct": 0.0,
                    "training_time_mins": 0.1,
                },
            },
        )

        logger.info(
            "TEST MODE: Job %s completed successfully.",
            job_id[:8]
        )

    except Exception as exc:

        logger.exception(
            "TEST MODE: Pipeline failed for job %s: %s",
            job_id[:8],
            exc
        )

        _set_status(
            job_store,
            job_id,
            "failed",
            error=str(exc)
        )


# =============================================================================
# REAL QLoRA PIPELINE
# =============================================================================

def run_pipeline(
    job_id: str,
    model_id: str,
    data_file_path: str,
    job_store: Dict[str, Any],
) -> None:
    """
    Full AdaptEval fine-tuning pipeline.

    When TEST_MODE = True:
        Uses run_test_pipeline().

    When TEST_MODE = False:
        Runs the real QLoRA pipeline.

    Real stages:

        queued
            ↓
        preparing
            ↓
        training
            ↓
        evaluating
            ↓
        complete

    Any error:
        ↓
        failed
    """

    # ── TEST MODE ─────────────────────────────────────────────────────

    if TEST_MODE:

        run_test_pipeline(
            job_id=job_id,
            model_id=model_id,
            data_file_path=data_file_path,
            job_store=job_store,
        )

        return

    # ── REAL PIPELINE ─────────────────────────────────────────────────

    start_time = time.time()

    try:

        # ── Load configurations ──────────────────────────────────────

        lora_cfg = _load_yaml(
            CONFIG_LORA
        )["lora"]

        training_cfg = _load_yaml(
            CONFIG_TRAINING
        )["training"]

        model_cfg = _load_yaml(
            CONFIG_TRAINING
        )["model"]

        adapter_dir = os.path.join(
            ADAPTER_BASE,
            f"adapter_{job_id}"
        )

        training_cfg["output_dir"] = (
            f"checkpoints/{job_id}"
        )

        # ── Stage 1: PREPARING ───────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "preparing"
        )

        # Prepare dataset splits from uploaded file.

        from data.prepare_dataset import main as prepare_data

        import data.prepare_dataset as prep_module

        # Point dataset preparation to the uploaded file.

        prep_module.RAW_FILE = data_file_path

        prep_module.OUTPUT_DIR = (
            f"data/processed/{job_id}/"
        )

        prepare_data()

        # Paths to generated datasets.

        processed_dir = (
            f"data/processed/{job_id}/"
        )

        training_cfg["train_file"] = (
            processed_dir + "train.jsonl"
        )

        training_cfg["val_file"] = (
            processed_dir + "val.jsonl"
        )

        training_cfg["test_file"] = (
            processed_dir + "test.jsonl"
        )

        # ── Load base model ──────────────────────────────────────────

        logger.info(
            "Loading base model: %s",
            model_id
        )

        model, tokenizer = load_base_model(
            model_name=model_id,
            load_in_4bit=model_cfg.get(
                "load_in_4bit",
                True
            ),
        )

        # ── Tokenize dataset ─────────────────────────────────────────

        tokenized = tokenize_dataset(
            train_file=training_cfg["train_file"],
            val_file=training_cfg["val_file"],
            test_file=training_cfg["test_file"],
            tokenizer=tokenizer,
            max_length=training_cfg.get(
                "max_seq_length",
                512
            ),
        )

        # ── Inject LoRA adapters ─────────────────────────────────────

        finetuned_model = inject_lora_adapters(
            model=model,
            lora_cfg=lora_cfg,
            load_in_4bit=model_cfg.get(
                "load_in_4bit",
                True
            ),
        )

        # ── Stage 2: TRAINING ────────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "training",
            progress={
                "current_epoch": 0,
                "total_epochs": training_cfg[
                    "num_train_epochs"
                ],
                "train_loss": 0.0,
                "eval_loss": 0.0,
                "elapsed_mins": 0.0,
            },
        )

        trainer = run_training(
            model=finetuned_model,
            tokenizer=tokenizer,
            tokenized_dataset=tokenized,
            training_cfg=training_cfg,
        )

        # Update progress from trainer log history.

        if trainer.state.log_history:

            last = trainer.state.log_history[-1]

            job_store[job_id]["progress"] = {
                "current_epoch": int(
                    trainer.state.epoch
                    or training_cfg["num_train_epochs"]
                ),

                "total_epochs": training_cfg[
                    "num_train_epochs"
                ],

                "train_loss": last.get(
                    "loss",
                    0.0
                ),

                "eval_loss": last.get(
                    "eval_loss",
                    0.0
                ),

                "elapsed_mins": (
                    time.time() - start_time
                ) / 60,
            }

        # ── Stage 3: EVALUATING ──────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "evaluating"
        )

        # Load fresh base model for baseline comparison.

        logger.info(
            "Loading baseline model for comparison evaluation..."
        )

        baseline_model, _ = load_base_model(
            model_name=model_id,
            load_in_4bit=model_cfg.get(
                "load_in_4bit",
                True
            ),
        )

        metrics = run_evaluation(
            baseline_model=baseline_model,
            finetuned_model=finetuned_model,
            tokenizer=tokenizer,
            test_dataset=tokenized["test"],
        )

        # ── Merge and save adapter ───────────────────────────────────

        merge_and_save(
            base_model=baseline_model,
            finetuned_model=finetuned_model,
            tokenizer=tokenizer,
            adapter_output_dir=adapter_dir,
        )

        # ── Adapter size ─────────────────────────────────────────────

        adapter_size_mb = sum(
            os.path.getsize(
                os.path.join(adapter_dir, f)
            )

            for f in os.listdir(adapter_dir)

            if os.path.isfile(
                os.path.join(adapter_dir, f)
            )
        ) / (1024 * 1024)

        # ── Trainable parameter percentage ───────────────────────────

        trainable = sum(
            p.numel()
            for p in finetuned_model.parameters()
            if p.requires_grad
        )

        total = sum(
            p.numel()
            for p in finetuned_model.parameters()
        )

        trained_pct = (
            100.0 * trainable / total
            if total > 0
            else 0.0
        )

        training_time_mins = (
            time.time() - start_time
        ) / 60

        # ── Sample predictions ───────────────────────────────────────

        samples = []

        preds_file = (
            "outputs/sample_predictions.jsonl"
        )

        if os.path.exists(preds_file):

            with open(
                preds_file,
                "r",
                encoding="utf-8"
            ) as f:

                samples = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]

        # ── Stage 4: COMPLETE ───────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "complete",
            results={
                "baseline": metrics["baseline"],

                "finetuned": metrics["finetuned"],

                "improvement": metrics["improvement"],

                "samples": samples,

                "adapter": {
                    "size_mb": round(
                        adapter_size_mb,
                        2
                    ),

                    "trained_params_pct": round(
                        trained_pct,
                        4
                    ),

                    "training_time_mins": round(
                        training_time_mins,
                        2
                    ),
                },
            },
        )

        logger.info(
            "Pipeline complete for job %s in %.1f minutes.",
            job_id[:8],
            training_time_mins,
        )

    except Exception as exc:

        logger.exception(
            "Pipeline failed for job %s: %s",
            job_id[:8],
            exc
        )

        _set_status(
            job_store,
            job_id,
            "failed",
            error=str(exc)
        )


if __name__ == "__main__":
    print(
        "AdaptEval pipeline module loaded."
    )

    print(
        f"TEST_MODE = {TEST_MODE}"
    )