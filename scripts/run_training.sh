#!/bin/bash
# run_training.sh
# Runs the full AdaptEval fine-tuning pipeline end-to-end.
# Usage: bash scripts/run_training.sh

set -euo pipefail   # exit on error, undefined variable, or pipe failure

echo "========================================="
echo " AdaptEval — Fine-Tuning Pipeline"
echo "========================================="

# Step 1: Validate environment
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Activate your virtual environment first."
    exit 1
fi

# Step 2: Prepare dataset
echo ""
echo "[1/3] Preparing dataset..."
python data/prepare_dataset.py

# Step 3: Run fine-tuning
echo ""
echo "[2/3] Running fine-tuning..."
python -m src.train \
    --lora_config configs/lora_config.yaml \
    --training_config configs/training_config.yaml

# Step 4: Run evaluation
echo ""
echo "[3/3] Running evaluation..."
python -m src.evaluate \
    --training_config configs/training_config.yaml

echo ""
echo "========================================="
echo " Pipeline complete."
echo " Metrics   → outputs/eval_metrics.json"
echo " Samples   → outputs/sample_predictions.jsonl"
echo " Adapter   → adapters/domain_adapter_v1/"
echo "========================================="