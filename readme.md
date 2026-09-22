# AdaptEval

An automated LoRA/QLoRA fine-tuning and evaluation framework for
domain-specific Large Language Model adaptation.

## What it does

- User selects a base LLM and uploads domain-specific training data
- System fine-tunes the model using LoRA — no manual ML setup required
- Returns a dashboard comparing baseline vs fine-tuned performance
- Produces a downloadable adapter weights file as the final artifact

## Project structure

lora-domain-finetune/
├── configs/ # LoRA and training hyperparameters
├── data/ # Raw data, processed splits, preparation script
├── src/ # Core pipeline modules
├── adapters/ # Saved adapter weights after training
├── checkpoints/ # Training checkpoints per epoch
├── outputs/ # Evaluation metrics and sample predictions
├── notebooks/ # Exploratory evaluation notebook
└── scripts/ # Shell scripts to run training and inference


## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Prepare data, fine-tune, and evaluate in one command
bash scripts/run_training.sh

# Run inference on the fine-tuned model
bash scripts/run_inference.sh "Your input prompt here"
```

## Output

| File | Description |
|---|---|
| `outputs/eval_metrics.json` | Accuracy, F1, perplexity for baseline and fine-tuned |
| `outputs/sample_predictions.jsonl` | Side-by-side sample outputs from both models |
| `adapters/domain_adapter_v1/` | Merged fine-tuned model weights |

## Hardware

Fine-tuning completes in 15–45 minutes on a T4 GPU (Google Colab)
for datasets up to 5,000 examples using 4-bit QLoRA.

On the UI — separate repo, start after backend is solid

Since this is already a sizable backend, I'd recommend the UI lives as a separate directory alongside this one rather than inside it:

adapteval/
├── backend/        ← everything above (lora-domain-finetune/)
└── frontend/       ← React app

The frontend needs just one backend API contract to start building against — a single FastAPI app with these endpoints:

Endpoint	Method	What it does
/models	GET	Returns list of supported base models
/finetune	POST	Accepts model name + uploaded .jsonl file, kicks off training
/status/{job_id}	GET	Returns training progress (epoch, loss, ETA)
/results/{job_id}	GET	Returns eval_metrics.json + sample_predictions.jsonl
/download/{job_id}	GET	Streams the adapter .safetensors file