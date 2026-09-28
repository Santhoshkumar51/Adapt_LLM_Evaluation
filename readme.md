# AdaptEval

AdaptEval is a web-based framework for parameter-efficient
fine-tuning and evaluation of large language models using
LoRA and QLoRA.

The system provides a unified workflow for:

- Base-model selection
- Dataset upload
- Dataset preprocessing
- LoRA/QLoRA fine-tuning
- Baseline vs fine-tuned evaluation
- ROUGE-L evaluation
- Perplexity evaluation
- Training-loss visualization
- Qualitative response comparison
- LoRA adapter generation
- Adapter download

---

## Supported Models

AdaptEval supports three base language models:

| Model | Parameters | Hugging Face ID |
|---|---:|---|
| Mistral 7B | 7B | `mistralai/Mistral-7B-v0.1` |
| Llama 3 8B | 8B | `meta-llama/Meta-Llama-3-8B` |
| Qwen 2 7B | 7B | `Qwen/Qwen2-7B` |

The experimental results reported in the accompanying paper
use **Mistral 7B**.

Llama 3 8B requires the appropriate Hugging Face access and
authentication because its repository is gated.

---

## Project Structure

```text
Adapt_LLM_Evaluation/
│
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   └── schemas.py
│   ├── main.py
│   ├── pipeline.py
│   └── store.py
│
├── configs/
│   ├── column_map.yaml
│   ├── lora_config.yaml
│   └── training_config.yaml
│
├── data/
│   └── prepare_dataset.py
│
├── frontend/
│   └── src/
│
├── src/
│   ├── evaluate.py
│   ├── inject_lora.py
│   ├── load_base_model.py
│   ├── merge_adapter.py
│   ├── tokenize_dataset.py
│   └── train.py
│
├── scripts/
│   ├── run_inference.sh
│   └── run_training.sh
│
├── adapters/
├── checkpoints/
└── outputs/