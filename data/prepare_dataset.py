"""
Reads raw domain Q&A data, formats it into instruction-response
prompt template, splits into train/val/test, and saves as .jsonl files.
"""

import json
import random
import os

RAW_FILE = "data/raw/domain_qa_raw.jsonl"
OUTPUT_DIR = "data/processed/"
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{response}"""


def format_example(example: dict) -> dict:
    return {
        "text": PROMPT_TEMPLATE.format(
            instruction=example["question"],
            response=example["answer"]
        )
    }


def split_data(data: list, ratios: dict) -> dict:
    random.shuffle(data)
    n = len(data)
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])
    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }


def save_split(split: list, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        for example in split:
            f.write(json.dumps(example) + "\n")
    print(f"Saved {len(split)} examples to {filename}")


def main():
    with open(RAW_FILE, "r") as f:
        raw_data = [json.loads(line) for line in f]

    formatted = [format_example(ex) for ex in raw_data]
    splits = split_data(formatted, SPLIT_RATIOS)

    for split_name, split_data in splits.items():
        save_split(split_data, f"{split_name}.jsonl")

    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()