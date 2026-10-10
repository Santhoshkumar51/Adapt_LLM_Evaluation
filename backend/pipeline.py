"""
AdaptEval pipeline.py

Connects FastAPI routes to the src/ training modules.
Updates JOB_STORE at every stage so the frontend polling
/status always sees the current state.

TEST_MODE = True  → safe local testing, no GPU or model needed
TEST_MODE = False → real QLoRA fine-tuning pipeline
"""

import gc
import json
import logging
import os
import time
from typing import Any, Dict, List

import torch
import yaml


logger = logging.getLogger(__name__)


# ── Config paths ──────────────────────────────────────────────────────────────

CONFIG_LORA = "configs/lora_config.yaml"
CONFIG_TRAINING = "configs/training_config.yaml"
ADAPTER_BASE = "adapters"


# ── Switch between local testing and real training ────────────────────────────

# True  → safe on any laptop, no GPU required, returns realistic dummy results
# False → real QLoRA pipeline, requires CUDA GPU
TEST_MODE = False


# =============================================================================
# Helpers
# =============================================================================

def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return as a dict."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def _set_status(
    job_store: Dict,
    job_id: str,
    status: str,
    **kwargs,
) -> None:
    """
    Update a job's status in the shared JOB_STORE dict.

    Any additional keyword arguments such as progress, results,
    or error are merged directly into the job's state entry.
    """

    job_store[job_id]["status"] = status

    for key, val in kwargs.items():
        job_store[job_id][key] = val

    logger.info(
        "Job %s → %s",
        job_id[:8],
        status,
    )


def _count_lines(path: str) -> int:
    """Return number of non-empty lines in a file."""

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return sum(
                1
                for line in f
                if line.strip()
            )

    except FileNotFoundError:

        return 0


def _clear_gpu_memory() -> None:
    """
    Release unused Python objects and CUDA cached memory.

    This is important on low-VRAM GPUs because AdaptEval evaluates
    the baseline and fine-tuned models sequentially.
    """

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =============================================================================
# TEST PIPELINE
# =============================================================================

def run_test_pipeline(
    job_id: str,
    model_id: str,
    data_file_path: str,
    job_store: Dict[str, Any],
) -> None:
    """
    Safe local development pipeline.

    Runs real dataset preparation on the uploaded file so the
    train/val/test splits are genuine and row counts are accurate.

    Model training and evaluation are simulated.
    """

    try:

        # ── Stage 1: PREPARING ─────────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "preparing",
        )

        logger.info(
            "TEST MODE: Preparing dataset for job %s",
            job_id[:8],
        )

        import data.prepare_dataset as prep_module

        prep_module.RAW_FILE = data_file_path
        prep_module.OUTPUT_DIR = (
            f"data/processed/{job_id}/"
        )

        prep_module.main()

        logger.info(
            "TEST MODE: Dataset preparation complete."
        )

        processed_dir = (
            f"data/processed/{job_id}/"
        )

        train_n = _count_lines(
            processed_dir + "train.jsonl"
        )

        val_n = _count_lines(
            processed_dir + "val.jsonl"
        )

        test_n = _count_lines(
            processed_dir + "test.jsonl"
        )

        total_n = (
            train_n
            + val_n
            + test_n
        )

        # ── Stage 2: TRAINING ──────────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "training",
            progress={
                "current_epoch": 1,
                "total_epochs": 3,
                "train_loss": 1.8420,
                "eval_loss": 1.6540,
                "elapsed_mins": 0.1,
            },
        )

        logger.info(
            "TEST MODE: Simulating training (3s sleep)."
        )

        time.sleep(3)

        # ── Stage 3: EVALUATING ────────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "evaluating",
        )

        logger.info(
            "TEST MODE: Simulating evaluation (3s sleep)."
        )

        time.sleep(3)

        # ── Stage 4: COMPLETE ──────────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "complete",
            results={

                "baseline": {
                    "rouge_l": 0.2134,
                    "perplexity": 45.3200,
                },

                "finetuned": {
                    "rouge_l": 0.4891,
                    "perplexity": 18.7400,
                },

                "improvement": {
                    "rouge_l_delta": 0.2757,
                    "perplexity_delta": 26.5800,
                },

                "loss_history": [
                    {
                        "epoch": 1,
                        "train_loss": 1.8420,
                        "eval_loss": 1.6540,
                    },
                    {
                        "epoch": 2,
                        "train_loss": 1.4210,
                        "eval_loss": 1.3120,
                    },
                    {
                        "epoch": 3,
                        "train_loss": 1.1030,
                        "eval_loss": 1.0870,
                    },
                ],

                "dataset_stats": {
                    "train": train_n,
                    "val": val_n,
                    "test": test_n,
                    "total": total_n,
                },

                "samples": [],

                "adapter": {
                    "size_mb": 18.4,
                    "trained_params_pct": 0.6420,
                    "training_time_mins": 0.1,
                },
            },
        )

        logger.info(
            "TEST MODE: Job %s completed. "
            "Dataset: %d total (%d/%d/%d).",
            job_id[:8],
            total_n,
            train_n,
            val_n,
            test_n,
        )

    except Exception as exc:

        logger.exception(
            "TEST MODE: Pipeline failed for job %s: %s",
            job_id[:8],
            exc,
        )

        _set_status(
            job_store,
            job_id,
            "failed",
            error=str(exc),
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

    TEST_MODE=True:
        Runs the safe test pipeline.

    TEST_MODE=False:
        Executes the real QLoRA pipeline.

    Real pipeline:

        queued
            ↓
        preparing
            ↓
        training
            ↓
        evaluating
            ↓
        complete

    Evaluation is performed sequentially to prevent multiple
    7B models from occupying the 8 GB GPU simultaneously.
    """

    # ── Test mode shortcut ─────────────────────────────────────────────

    if TEST_MODE:

        run_test_pipeline(
            job_id=job_id,
            model_id=model_id,
            data_file_path=data_file_path,
            job_store=job_store,
        )

        return

    # ── Real QLoRA pipeline ────────────────────────────────────────────

    # Heavy imports are kept inside the real pipeline so TEST_MODE
    # does not require the ML stack.

    from src.load_base_model import load_base_model

    from src.tokenize_dataset import (
        tokenize_dataset,
    )

    from src.inject_lora import (
        inject_lora_adapters,
    )

    from src.train import (
        run_training,
    )

    from src.evaluate import (
        evaluate_single_model,
        save_evaluation_results,
    )

    from src.merge_adapter import (
        merge_and_save,
    )

    start_time = time.time()

    try:

        # ── Load configurations ───────────────────────────────────────

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
            f"adapter_{job_id}",
        )

        training_cfg["output_dir"] = (
            f"checkpoints/{job_id}"
        )

        # ── Stage 1: PREPARING ─────────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "preparing",
        )

        import data.prepare_dataset as prep_module

        prep_module.RAW_FILE = data_file_path

        prep_module.OUTPUT_DIR = (
            f"data/processed/{job_id}/"
        )

        prep_module.main()

        processed_dir = (
            f"data/processed/{job_id}/"
        )

        training_cfg["train_file"] = (
            processed_dir
            + "train.jsonl"
        )

        training_cfg["val_file"] = (
            processed_dir
            + "val.jsonl"
        )

        training_cfg["test_file"] = (
            processed_dir
            + "test.jsonl"
        )

        # ── Load base model ────────────────────────────────────────────

        logger.info(
            "Loading base model: %s",
            model_id,
        )

        model, tokenizer = load_base_model(
            model_name=model_id,
            load_in_4bit=model_cfg.get(
                "load_in_4bit",
                True,
            ),
        )

        # ── Tokenize dataset ───────────────────────────────────────────

        tokenized = tokenize_dataset(
            train_file=training_cfg[
                "train_file"
            ],
            val_file=training_cfg[
                "val_file"
            ],
            test_file=training_cfg[
                "test_file"
            ],
            tokenizer=tokenizer,
            max_length=training_cfg.get(
                "max_seq_length",
                512,
            ),
        )

        # Real dataset split sizes

        dataset_stats = {
            "train": len(
                tokenized["train"]
            ),

            "val": len(
                tokenized["val"]
            ),

            "test": len(
                tokenized["test"]
            ),

            "total": (
                len(tokenized["train"])
                + len(tokenized["val"])
                + len(tokenized["test"])
            ),
        }

        logger.info(
            "Split sizes — Train: %d | Val: %d | Test: %d",
            dataset_stats["train"],
            dataset_stats["val"],
            dataset_stats["test"],
        )

        # ── Inject LoRA adapters ──────────────────────────────────────

        finetuned_model = inject_lora_adapters(
            model=model,
            lora_cfg=lora_cfg,
            load_in_4bit=model_cfg.get(
                "load_in_4bit",
                True,
            ),
        )

        # ── Stage 2: TRAINING ──────────────────────────────────────────

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

        # ── Extract per-epoch loss curve ───────────────────────────────

        epoch_train: Dict[int, float] = {}
        epoch_eval: Dict[int, float] = {}

        for entry in trainer.state.log_history:

            ep = int(
                entry.get(
                    "epoch",
                    0,
                )
            )

            if (
                "loss" in entry
                and "eval_loss" not in entry
            ):

                epoch_train[ep] = (
                    entry["loss"]
                )

            if "eval_loss" in entry:

                epoch_eval[ep] = (
                    entry["eval_loss"]
                )

        loss_history: List[Dict] = []

        all_epochs = sorted(
            set(
                list(
                    epoch_train.keys()
                )
                + list(
                    epoch_eval.keys()
                )
            )
        )

        for ep in all_epochs:

            loss_history.append(
                {
                    "epoch": ep,

                    "train_loss": round(
                        epoch_train.get(
                            ep,
                            0.0,
                        ),
                        4,
                    ),

                    "eval_loss": round(
                        epoch_eval.get(
                            ep,
                            0.0,
                        ),
                        4,
                    ),
                }
            )

        # Update final training progress

        if trainer.state.log_history:

            last_train_loss = 0.0
            last_eval_loss = 0.0

            for entry in reversed(
                trainer.state.log_history
            ):

                if (
                    last_eval_loss == 0.0
                    and "eval_loss" in entry
                ):

                    last_eval_loss = entry[
                        "eval_loss"
                    ]

                if (
                    last_train_loss == 0.0
                    and "loss" in entry
                    and "eval_loss" not in entry
                ):

                    last_train_loss = entry[
                        "loss"
                    ]

                if (
                    last_train_loss != 0.0
                    and last_eval_loss != 0.0
                ):

                    break

            job_store[job_id][
                "progress"
            ] = {

                "current_epoch": int(
                    trainer.state.epoch
                    or training_cfg[
                        "num_train_epochs"
                    ]
                ),

                "total_epochs": training_cfg[
                    "num_train_epochs"
                ],

                "train_loss": last_train_loss,

                "eval_loss": last_eval_loss,

                "elapsed_mins": (
                    time.time()
                    - start_time
                ) / 60,
            }

        # ── Save LoRA adapter ──────────────────────────────────────────
        #
        # The trained model is saved as a separate LoRA adapter.
        # It is NOT merged into the base model.

        logger.info(
            "Saving trained LoRA adapter..."
        )

        os.makedirs(
            adapter_dir,
            exist_ok=True,
        )

        merge_and_save(
            base_model=finetuned_model,
            finetuned_model=finetuned_model,
            tokenizer=tokenizer,
            adapter_output_dir=adapter_dir,
        )

        logger.info(
            "LoRA adapter saved to: %s",
            adapter_dir,
        )

        # ── Trainable parameter percentage ─────────────────────────────

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

        # ── Release training model ────────────────────────────────────
        #
        # This is critical for the RTX 4060 8 GB GPU.
        # We must release the trained model before loading the
        # baseline model.

        logger.info(
            "Releasing fine-tuned model from GPU "
            "before baseline evaluation..."
        )

        try:

            trainer.model = None

        except Exception:

            pass

        del trainer
        del finetuned_model
        del model

        _clear_gpu_memory()

        logger.info(
            "GPU memory released."
        )

        # ── Stage 3: EVALUATING ────────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "evaluating",
        )

        # ==============================================================
        # BASELINE EVALUATION
        # ==============================================================

        logger.info(
            "Loading baseline model for evaluation..."
        )

        baseline_model, baseline_tokenizer = (
            load_base_model(
                model_name=model_id,
                load_in_4bit=model_cfg.get(
                    "load_in_4bit",
                    True,
                ),
            )
        )

        baseline_result = evaluate_single_model(
            model=baseline_model,
            tokenizer=baseline_tokenizer,
            test_dataset=tokenized["test"],
            model_label="baseline",
        )

        logger.info(
            "Baseline evaluation complete."
        )

        # Release baseline model

        del baseline_model
        del baseline_tokenizer

        _clear_gpu_memory()

        logger.info(
            "Baseline model released from GPU."
        )

        # ==============================================================
        # FINE-TUNED MODEL EVALUATION
        #
        # Load the base model again and attach the saved LoRA adapter.
        # Only one 7B model is present on the GPU.
        # ==============================================================

        logger.info(
            "Loading base model for fine-tuned evaluation..."
        )

        finetuned_base_model, finetuned_tokenizer = (
            load_base_model(
                model_name=model_id,
                load_in_4bit=model_cfg.get(
                    "load_in_4bit",
                    True,
                ),
            )
        )

        from peft import PeftModel

        logger.info(
            "Loading saved LoRA adapter from: %s",
            adapter_dir,
        )

        finetuned_eval_model = (
            PeftModel.from_pretrained(
                finetuned_base_model,
                adapter_dir,
                is_trainable=False,
            )
        )

        finetuned_result = evaluate_single_model(
            model=finetuned_eval_model,
            tokenizer=finetuned_tokenizer,
            test_dataset=tokenized["test"],
            model_label="fine-tuned",
        )

        logger.info(
            "Fine-tuned evaluation complete."
        )

        # ── Combine evaluation results ────────────────────────────────

        metrics = save_evaluation_results(
            baseline_result=baseline_result,
            finetuned_result=finetuned_result,
        )

        # Release evaluation model

        del finetuned_eval_model
        del finetuned_base_model
        del finetuned_tokenizer

        _clear_gpu_memory()

        logger.info(
            "Evaluation models released from GPU."
        )

        # ── Adapter file size ─────────────────────────────────────────

        adapter_size_mb = sum(
            os.path.getsize(
                os.path.join(
                    adapter_dir,
                    f,
                )
            )

            for f in os.listdir(
                adapter_dir
            )

            if os.path.isfile(
                os.path.join(
                    adapter_dir,
                    f,
                )
            )

        ) / (
            1024 * 1024
        )

        # ── Training time ─────────────────────────────────────────────

        training_time_mins = (
            time.time()
            - start_time
        ) / 60

        # ── Load sample predictions ───────────────────────────────────

        samples = []

        preds_file = (
            "outputs/sample_predictions.jsonl"
        )

        if os.path.exists(
            preds_file
        ):

            with open(
                preds_file,
                "r",
                encoding="utf-8",
            ) as f:

                samples = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]

        # ── Stage 4: COMPLETE ─────────────────────────────────────────

        _set_status(
            job_store,
            job_id,
            "complete",
            results={

                "baseline":
                    metrics["baseline"],

                "finetuned":
                    metrics["finetuned"],

                "improvement":
                    metrics["improvement"],

                "loss_history":
                    loss_history,

                "dataset_stats":
                    dataset_stats,

                "samples":
                    samples,

                "adapter": {

                    "size_mb":
                        round(
                            adapter_size_mb,
                            2,
                        ),

                    "trained_params_pct":
                        round(
                            trained_pct,
                            4,
                        ),

                    "training_time_mins":
                        round(
                            training_time_mins,
                            2,
                        ),
                },
            },
        )

        logger.info(
            "Pipeline complete for job %s in %.1f minutes. "
            "ROUGE-L: %.4f → %.4f | "
            "Perplexity: %.2f → %.2f",

            job_id[:8],

            training_time_mins,

            metrics["baseline"][
                "rouge_l"
            ],

            metrics["finetuned"][
                "rouge_l"
            ],

            metrics["baseline"][
                "perplexity"
            ],

            metrics["finetuned"][
                "perplexity"
            ],
        )

    except Exception as exc:

        logger.exception(
            "Pipeline failed for job %s: %s",
            job_id[:8],
            exc,
        )

        _clear_gpu_memory()

        _set_status(
            job_store,
            job_id,
            "failed",
            error=str(exc),
        )


# =============================================================================
# Direct execution
# =============================================================================

if __name__ == "__main__":

    print(
        f"AdaptEval pipeline loaded. "
        f"TEST_MODE = {TEST_MODE}"
    )