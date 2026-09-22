#!/bin/bash
# run_inference.sh
# Runs a single inference call against the fine-tuned merged model.
# Usage: bash scripts/run_inference.sh "Your input prompt here"

set -euo pipefail

PROMPT="${1:-}"
ADAPTER_DIR="adapters/domain_adapter_v1/"

if [ -z "$PROMPT" ]; then
    echo "ERROR: No prompt provided."
    echo "Usage: bash scripts/run_inference.sh \"Your input prompt here\""
    exit 1
fi

if [ ! -d "$ADAPTER_DIR" ]; then
    echo "ERROR: Adapter directory not found at $ADAPTER_DIR"
    echo "Run scripts/run_training.sh first."
    exit 1
fi

echo "Running inference with fine-tuned adapter..."
echo "Prompt: $PROMPT"
echo ""

python -c "
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

adapter_dir = '$ADAPTER_DIR'
prompt = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
model = AutoModelForCausalLM.from_pretrained(
    adapter_dir,
    torch_dtype=torch.float16,
    device_map='auto',
)

inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)

response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print('Response:', response)
" "$PROMPT"