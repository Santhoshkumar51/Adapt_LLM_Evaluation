"""
AdaptEval pipeline.py

Connects FastAPI routes to the src/ training modules.
Updates JOB_STORE at every stage so the frontend polling
/status always sees the current state.

TEST_MODE = True  → safe local testing, no GPU or model needed
TEST_MODE = False → real QLoRA fine-tuning pipeline
"""

import json
import logging
import os
import time
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

# ── Config paths ──────────────────────────────────────────────────────────────
CONFIG_LORA     = "configs/lora_config.yaml"
CONFIG_TRAINING = "configs/training_config.yaml"
ADAPTER_BASE    = "adapters"

# ── Switch between local testing and real training ────────────────────────────
# True  → safe on any laptop, no GPU required, returns realistic dummy results
# False → real QLoRA pipeline, requires CUDA GPU (Colab T4 or better)
TEST_MODE = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_status(
    job_store: Dict,
    job_id: str,
    status: str,
    **kwargs,
) -> None:
    """
    Update a job's status in the shared JOB_STORE dict.
    Any additional keyword arguments (progress, results, error)
    are merged directly into the job's state entry.
    """
    job_store[job_id]["status"] = status
    for key, val in kwargs.items():
        job_store[job_id][key] = val
    logger.info("Job %s → %s", job_id[:8], status)


def _count_lines(path: str) -> int:
    """Return number of non-empty lines in a file. Returns 0 if not found."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


# =============================================================================
# TEST PIPELINE
# Verifies: upload → dataset prep → job status updates → frontend polling
# Does NOT load any model or require a GPU.
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
    All model training and evaluation steps are simulated with
    realistic dummy values so the full dashboard renders correctly.

    Args:
        job_id:         UUID assigned at job submission.
        model_id:       Selected Hugging Face model ID (unused in test mode).
        data_file_path: Path to the uploaded .jsonl or .csv dataset file.
        job_store:      Shared in-memory job state dict from store.py.
    """
    try:

        # ── Stage 1: PREPARING ────────────────────────────────────────
        _set_status(job_store, job_id, "preparing")

        logger.info("TEST MODE: Preparing dataset for job %s", job_id[:8])

        import data.prepare_dataset as prep_module

        prep_module.RAW_FILE   = data_file_path
        prep_module.OUTPUT_DIR = f"data/processed/{job_id}/"

        # Run real dataset preparation — splits are genuine
        prep_module.main()

        logger.info("TEST MODE: Dataset preparation complete.")

        # Count actual split sizes from the generated files
        processed_dir = f"data/processed/{job_id}/"
        train_n = _count_lines(processed_dir + "train.jsonl")
        val_n   = _count_lines(processed_dir + "val.jsonl")
        test_n  = _count_lines(processed_dir + "test.jsonl")
        total_n = train_n + val_n + test_n

        # ── Stage 2: TRAINING ─────────────────────────────────────────
        _set_status(
            job_store, job_id, "training",
            progress={
                "current_epoch": 1,
                "total_epochs":  3,
                "train_loss":    1.8420,
                "eval_loss":     1.6540,
                "elapsed_mins":  0.1,
            },
        )

        logger.info("TEST MODE: Simulating training (3s sleep).")
        time.sleep(3)

        # ── Stage 3: EVALUATING ───────────────────────────────────────
        _set_status(job_store, job_id, "evaluating")

        logger.info("TEST MODE: Simulating evaluation (3s sleep).")
        time.sleep(3)

        # ── Stage 4: COMPLETE ─────────────────────────────────────────
        _set_status(
            job_store, job_id, "complete",
            results={
                # Realistic dummy metrics — match schema fields exactly
                "baseline": {
                    "rouge_l":    0.2134,
                    "perplexity": 45.3200,
                },
                "finetuned": {
                    "rouge_l":    0.4891,
                    "perplexity": 18.7400,
                },
                "improvement": {
                    "rouge_l_delta":    0.2757,
                    "perplexity_delta": 26.5800,
                },

                # Simulated loss curve across 3 epochs
                "loss_history": [
                    {"epoch": 1, "train_loss": 1.8420, "eval_loss": 1.6540},
                    {"epoch": 2, "train_loss": 1.4210, "eval_loss": 1.3120},
                    {"epoch": 3, "train_loss": 1.1030, "eval_loss": 1.0870},
                ],

                # Real split sizes from the actual uploaded file
                "dataset_stats": {
                    "train": train_n,
                    "val":   val_n,
                    "test":  test_n,
                    "total": total_n,
                },

                # No sample predictions in test mode
                "samples": [],

                # Realistic adapter metadata
                "adapter": {
                    "size_mb":            18.4,
                    "trained_params_pct": 0.6420,
                    "training_time_mins": 0.1,
                },
            },
        )

        logger.info(
            "TEST MODE: Job %s completed. Dataset: %d total (%d/%d/%d).",
            job_id[:8], total_n, train_n, val_n, test_n,
        )

    except Exception as exc:
        logger.exception(
            "TEST MODE: Pipeline failed for job %s: %s",
            job_id[:8], exc,
        )
        _set_status(job_store, job_id, "failed", error=str(exc))


# =============================================================================
# REAL QLoRA PIPELINE
# Requires a CUDA GPU. Run on Google Colab T4 or better.
# Set TEST_MODE = False before running.
# =============================================================================

def run_pipeline(
    job_id: str,
    model_id: str,
    data_file_path: str,
    job_store: Dict[str, Any],
) -> None:
    """
    Full AdaptEval fine-tuning pipeline.

    Routes to run_test_pipeline() when TEST_MODE is True.
    When TEST_MODE is False, executes the real QLoRA pipeline:

        queued → preparing → training → evaluating → complete
                                                   ↘ failed (on error)

    Args:
        job_id:         UUID assigned at job submission.
        model_id:       Hugging Face model ID selected by the user.
        data_file_path: Path to the uploaded dataset file on disk.
        job_store:      Shared in-memory job state dict from store.py.
    """

    # ── Test mode shortcut ────────────────────────────────────────────
    if TEST_MODE:
        run_test_pipeline(
            job_id=job_id,
            model_id=model_id,
            data_file_path=data_file_path,
            job_store=job_store,
        )
        return

    # ── Real QLoRA pipeline ───────────────────────────────────────────
    # Import src modules here so test mode never triggers their
    # heavy imports (torch, transformers, peft, bitsandbytes).
    from src.load_base_model  import load_base_model
    from src.tokenize_dataset import tokenize_dataset
    from src.inject_lora      import inject_lora_adapters
    from src.train            import run_training
    from src.evaluate         import run_evaluation
    from src.merge_adapter    import merge_and_save

    start_time = time.time()

    try:

        # ── Load configs ──────────────────────────────────────────────
        lora_cfg     = _load_yaml(CONFIG_LORA)["lora"]
        training_cfg = _load_yaml(CONFIG_TRAINING)["training"]
        model_cfg    = _load_yaml(CONFIG_TRAINING)["model"]

        adapter_dir = os.path.join(ADAPTER_BASE, f"adapter_{job_id}")
        training_cfg["output_dir"] = f"checkpoints/{job_id}"

        # ── Stage 1: PREPARING ────────────────────────────────────────
        _set_status(job_store, job_id, "preparing")

        import data.prepare_dataset as prep_module

        prep_module.RAW_FILE   = data_file_path
        prep_module.OUTPUT_DIR = f"data/processed/{job_id}/"
        prep_module.main()

        processed_dir = f"data/processed/{job_id}/"
        training_cfg["train_file"] = processed_dir + "train.jsonl"
        training_cfg["val_file"]   = processed_dir + "val.jsonl"
        training_cfg["test_file"]  = processed_dir + "test.jsonl"

        # ── Load base model ───────────────────────────────────────────
        logger.info("Loading base model: %s", model_id)

        model, tokenizer = load_base_model(
            model_name=model_id,
            load_in_4bit=model_cfg.get("load_in_4bit", True),
        )

        # ── Tokenize dataset ──────────────────────────────────────────
        tokenized = tokenize_dataset(
            train_file=training_cfg["train_file"],
            val_file=training_cfg["val_file"],
            test_file=training_cfg["test_file"],
            tokenizer=tokenizer,
            max_length=training_cfg.get("max_seq_length", 512),
        )

        # Real dataset split sizes from tokenized lengths
        dataset_stats = {
            "train": len(tokenized["train"]),
            "val":   len(tokenized["val"]),
            "test":  len(tokenized["test"]),
            "total": len(tokenized["train"]) + len(tokenized["val"]) + len(tokenized["test"]),
        }

        # ── Inject LoRA adapters ──────────────────────────────────────
        finetuned_model = inject_lora_adapters(
            model=model,
            lora_cfg=lora_cfg,
            load_in_4bit=model_cfg.get("load_in_4bit", True),
        )

        # ── Stage 2: TRAINING ─────────────────────────────────────────
        _set_status(
            job_store, job_id, "training",
            progress={
                "current_epoch": 0,
                "total_epochs":  training_cfg["num_train_epochs"],
                "train_loss":    0.0,
                "eval_loss":     0.0,
                "elapsed_mins":  0.0,
            },
        )

        trainer = run_training(
            model=finetuned_model,
            tokenizer=tokenizer,
            tokenized_dataset=tokenized,
            training_cfg=training_cfg,
        )

        # ── Extract per-epoch loss curve from trainer log ─────────────
        # trainer.state.log_history contains dicts like:
        #   {"loss": 1.84, "epoch": 1.0, ...}  ← training step entry
        #   {"eval_loss": 1.65, "epoch": 1.0, ...} ← eval entry
        # We bucket them by integer epoch and pair them up.
        epoch_train: Dict[int, float] = {}
        epoch_eval:  Dict[int, float] = {}

        for entry in trainer.state.log_history:
            ep = int(entry.get("epoch", 0))
            if "loss" in entry and "eval_loss" not in entry:
                epoch_train[ep] = entry["loss"]
            if "eval_loss" in entry:
                epoch_eval[ep] = entry["eval_loss"]

        loss_history: List[Dict] = []
        all_epochs = sorted(
            set(list(epoch_train.keys()) + list(epoch_eval.keys()))
        )
        for ep in all_epochs:
            loss_history.append({
                "epoch":      ep,
                "train_loss": round(epoch_train.get(ep, 0.0), 4),
                "eval_loss":  round(epoch_eval.get(ep, 0.0), 4),
            })

        # Update final progress snapshot
        if trainer.state.log_history:
            last = trainer.state.log_history[-1]
            job_store[job_id]["progress"] = {
                "current_epoch": int(
                    trainer.state.epoch or training_cfg["num_train_epochs"]
                ),
                "total_epochs":  training_cfg["num_train_epochs"],
                "train_loss":    last.get("loss", 0.0),
                "eval_loss":     last.get("eval_loss", 0.0),
                "elapsed_mins":  (time.time() - start_time) / 60,
            }

        # ── Stage 3: EVALUATING ───────────────────────────────────────
        _set_status(job_store, job_id, "evaluating")

        # Load a fresh copy of the base model for fair baseline comparison
        # (finetuned_model already has LoRA weights merged in memory)
        logger.info("Loading baseline model for comparison evaluation...")

        baseline_model, _ = load_base_model(
            model_name=model_id,
            load_in_4bit=model_cfg.get("load_in_4bit", True),
        )

        metrics = run_evaluation(
            baseline_model=baseline_model,
            finetuned_model=finetuned_model,
            tokenizer=tokenizer,
            test_dataset=tokenized["test"],
        )

        # ── Merge and save adapter ────────────────────────────────────
        merge_and_save(
            base_model=baseline_model,
            finetuned_model=finetuned_model,
            tokenizer=tokenizer,
            adapter_output_dir=adapter_dir,
        )

        # ── Adapter file size ─────────────────────────────────────────
        adapter_size_mb = sum(
            os.path.getsize(os.path.join(adapter_dir, f))
            for f in os.listdir(adapter_dir)
            if os.path.isfile(os.path.join(adapter_dir, f))
        ) / (1024 * 1024)

        # ── Trainable parameter percentage ────────────────────────────
        trainable = sum(
            p.numel() for p in finetuned_model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in finetuned_model.parameters())
        trained_pct = 100.0 * trainable / total if total > 0 else 0.0

        training_time_mins = (time.time() - start_time) / 60

        # ── Load sample predictions written by evaluate.py ────────────
        samples = []
        preds_file = "outputs/sample_predictions.jsonl"
        if os.path.exists(preds_file):
            with open(preds_file, "r", encoding="utf-8") as f:
                samples = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]

        # ── Stage 4: COMPLETE ─────────────────────────────────────────
        _set_status(
            job_store, job_id, "complete",
            results={
                "baseline":    metrics["baseline"],
                "finetuned":   metrics["finetuned"],
                "improvement": metrics["improvement"],
                "loss_history":  loss_history,
                "dataset_stats": dataset_stats,
                "samples": samples,
                "adapter": {
                    "size_mb":            round(adapter_size_mb, 2),
                    "trained_params_pct": round(trained_pct, 4),
                    "training_time_mins": round(training_time_mins, 2),
                },
            },
        )

        logger.info(
            "Pipeline complete for job %s in %.1f minutes. "
            "ROUGE-L: %.4f → %.4f | Perplexity: %.2f → %.2f",
            job_id[:8],
            training_time_mins,
            metrics["baseline"]["rouge_l"],
            metrics["finetuned"]["rouge_l"],
            metrics["baseline"]["perplexity"],
            metrics["finetuned"]["perplexity"],
        )

    except Exception as exc:
        logger.exception(
            "Pipeline failed for job %s: %s",
            job_id[:8], exc,
        )
        _set_status(job_store, job_id, "failed", error=str(exc))


if __name__ == "__main__":
    print(f"AdaptEval pipeline loaded. TEST_MODE = {TEST_MODE}")