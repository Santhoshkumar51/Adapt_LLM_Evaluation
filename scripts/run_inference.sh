#!/bin/bash

# AdaptEval — LoRA adapter inference
#
# Usage:
#   ADAPTER_DIR=adapters/adapter_JOB_ID \
#   bash scripts/run_inference.sh "Your input prompt"
#
# Optional:
#   BASE_MODEL=Qwen/Qwen2-7B \
#   ADAPTER_DIR=adapters/adapter_JOB_ID \
#   bash scripts/run_inference.sh "Your input prompt"

set -euo pipefail


PROMPT="${1:-}"

BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-v0.1}"

ADAPTER_DIR="${ADAPTER_DIR:-}"


if [ -z "$PROMPT" ]; then
    echo "ERROR: No prompt provided."
    echo ""
    echo "Usage:"
    echo 'ADAPTER_DIR=adapters/adapter_JOB_ID bash scripts/run_inference.sh "Your prompt"'
    exit 1
fi


if [ -z "$ADAPTER_DIR" ]; then
    echo "ERROR: ADAPTER_DIR is not set."
    exit 1
fi


if [ ! -d "$ADAPTER_DIR" ]; then
    echo "ERROR: Adapter directory not found:"
    echo "$ADAPTER_DIR"
    exit 1
fi


if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
    echo "ERROR: adapter_model.safetensors not found."
    exit 1
fi


echo "========================================="
echo " AdaptEval — LoRA Adapter Inference"
echo "========================================="
echo "Base model : $BASE_MODEL"
echo "Adapter    : $ADAPTER_DIR"
echo ""


python - "$PROMPT" "$BASE_MODEL" "$ADAPTER_DIR" <<'PY'

import sys

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


prompt = sys.argv[1]
base_model_name = sys.argv[2]
adapter_dir = sys.argv[3]


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


tokenizer = AutoTokenizer.from_pretrained(
    adapter_dir,
    trust_remote_code=True,
)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)


model = PeftModel.from_pretrained(
    base_model,
    adapter_dir,
)


model.eval()


inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)


with torch.no_grad():

    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )


generated_ids = output_ids[
    0
][
    inputs["input_ids"].shape[1]:
]


response = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
).strip()


print("Response:")
print(response)

PY