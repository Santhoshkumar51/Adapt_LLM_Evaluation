#!/bin/bash

# AdaptEval — run the complete pipeline directly.
#
# Usage:
#   bash scripts/run_training.sh data/raw/domain_qa_raw.csv
#
# Optional model:
#   MODEL_ID=mistralai/Mistral-7B-v0.1 \
#   bash scripts/run_training.sh data/raw/domain_qa_raw.csv

set -euo pipefail


DATA_FILE="${1:-}"

MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-v0.1}"


if [ -z "$DATA_FILE" ]; then
    echo "ERROR: No training dataset supplied."
    echo ""
    echo "Usage:"
    echo "bash scripts/run_training.sh <dataset.csv|dataset.jsonl>"
    exit 1
fi


if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Dataset not found:"
    echo "$DATA_FILE"
    exit 1
fi


echo "========================================="
echo " AdaptEval — Fine-Tuning Pipeline"
echo "========================================="
echo "Model : $MODEL_ID"
echo "Data  : $DATA_FILE"
echo ""


python - "$DATA_FILE" "$MODEL_ID" <<'PY'

import sys
import uuid

from backend.pipeline import run_pipeline
from backend.store import JOB_STORE
from backend.api.schemas import JobStatus


data_file = sys.argv[1]
model_id = sys.argv[2]

job_id = str(uuid.uuid4())


JOB_STORE[job_id] = {
    "status": JobStatus.QUEUED,
    "model_id": model_id,
    "progress": None,
    "results": None,
    "error": None,
}


print(f"Job ID: {job_id}")
print("Starting AdaptEval pipeline...")
print("")


run_pipeline(
    job_id=job_id,
    model_id=model_id,
    data_file_path=data_file,
    job_store=JOB_STORE,
)


job = JOB_STORE[job_id]


print("")
print("=========================================")
print(" Pipeline finished")
print("=========================================")
print(f"Status: {job['status']}")


if job.get("error"):
    print(f"Error: {job['error']}")
    raise SystemExit(1)

PY