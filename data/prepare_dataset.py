"""
Reads raw domain Q&A data, automatically detects common column names,
formats it into an instruction-response prompt template, splits into
train/val/test, and saves as .jsonl files.
"""

import json
import random
import os
import csv
import yaml


RAW_FILE = "data/raw/domain_qa_raw.csv"
OUTPUT_DIR = "data/processed/"
COLUMN_MAP_FILE = "configs/column_map.yaml"

SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{response}"""

MAX_SAMPLES = 5000
RANDOM_SEED = 42

def load_column_map():
    """Load supported column aliases from YAML."""

    with open(COLUMN_MAP_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_column(columns, aliases):
    """
    Find the actual dataset column using the aliases
    defined in column_map.yaml.
    """

    # Exact match first
    for alias in aliases:
        if alias in columns:
            return alias

    # Case-insensitive match
    lower_columns = {
        str(col).strip().lower(): col
        for col in columns
    }

    for alias in aliases:
        if alias.lower() in lower_columns:
            return lower_columns[alias.lower()]

    return None


def detect_columns(data, column_map):
    """Detect question, answer, context and category columns."""

    if not data:
        raise ValueError("Dataset is empty.")

    columns = list(data[0].keys())

    question_col = find_column(
        columns,
        column_map.get("question", [])
    )

    answer_col = find_column(
        columns,
        column_map.get("answer", [])
    )

    context_col = find_column(
        columns,
        column_map.get("context", [])
    )

    category_col = find_column(
        columns,
        column_map.get("category", [])
    )

    if question_col is None:
        raise ValueError(
            f"Could not detect a question/instruction column.\n"
            f"Available columns: {columns}"
        )

    if answer_col is None:
        raise ValueError(
            f"Could not detect an answer/response/output column.\n"
            f"Available columns: {columns}"
        )

    print("\nDetected columns:")
    print(f"  Question/Input : {question_col}")
    print(f"  Answer/Output  : {answer_col}")
    print(f"  Context        : {context_col}")
    print(f"  Category       : {category_col}")

    return {
        "question": question_col,
        "answer": answer_col,
        "context": context_col,
        "category": category_col
    }


def format_example(example: dict, columns: dict) -> dict:
    """Convert one raw example into the canonical training format."""

    question = str(example[columns["question"]]).strip()
    answer = str(example[columns["answer"]]).strip()

    context = None

    if columns["context"]:
        value = example.get(columns["context"])

        if value is not None and str(value).strip():
            context = str(value).strip()

    # If context exists, include it in the instruction.
    if context:
        instruction = (
            f"{question}\n\n"
            f"Context:\n{context}"
        )
    else:
        instruction = question

    formatted = {
        "text": PROMPT_TEMPLATE.format(
            instruction=instruction,
            response=answer
        )
    }

    # Preserve category as metadata if available.
    if columns["category"]:
        category = example.get(columns["category"])

        if category is not None and str(category).strip():
            formatted["category"] = str(category).strip()

    return formatted


def load_raw_data():
    """Load JSONL or CSV dataset."""

    extension = os.path.splitext(RAW_FILE)[1].lower()

    if extension == ".jsonl":
        with open(RAW_FILE, "r", encoding="utf-8") as f:
            return [
                json.loads(line)
                for line in f
                if line.strip()
            ]

    elif extension == ".json":
        with open(RAW_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    elif extension == ".csv":
        with open(
            RAW_FILE,
            "r",
            encoding="utf-8",
            newline=""
        ) as f:
            return list(csv.DictReader(f))

    else:
        raise ValueError(
            "Unsupported file format. "
            "Use .jsonl, .json, or .csv."
        )

def sample_data(data, columns, max_samples=5000):
    """
    Select at most max_samples while preserving category distribution
    when a category column is available.
    """

    if len(data) <= max_samples:
        return data

    category_col = columns.get("category")

    # No category → simple random sampling
    if not category_col:
        random.seed(RANDOM_SEED)
        return random.sample(data, max_samples)

    # Group examples by category
    groups = {}

    for example in data:
        category = str(
            example.get(category_col, "unknown")
        ).strip()

        if not category:
            category = "unknown"

        groups.setdefault(category, []).append(example)

    # Calculate proportional allocation
    total = len(data)

    allocations = {}

    for category, examples in groups.items():
        proportion = len(examples) / total
        allocations[category] = int(
            proportion * max_samples
        )

    # Make sure we don't lose samples due to rounding
    allocated = sum(allocations.values())
    remaining = max_samples - allocated

    # Give remaining slots to largest groups
    sorted_categories = sorted(
        groups.keys(),
        key=lambda c: len(groups[c]),
        reverse=True
    )

    for category in sorted_categories[:remaining]:
        allocations[category] += 1

    # Sample
    random.seed(RANDOM_SEED)

    sampled = []

    for category, examples in groups.items():
        count = min(
            allocations[category],
            len(examples)
        )

        sampled.extend(
            random.sample(examples, count)
        )

    random.shuffle(sampled)

    return sampled

def split_data(data: list, ratios: dict) -> dict:
    """Randomly split data into train/validation/test sets."""

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
    """Save a dataset split as JSONL."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        for example in split:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                ) + "\n"
            )

    print(
        f"Saved {len(split)} examples to {output_path}"
    )

def main():
    print("Loading dataset...")

    raw_data = load_raw_data()

    print(f"Loaded {len(raw_data)} examples.")

    column_map = load_column_map()

    columns = detect_columns(
        raw_data,
        column_map
    )

    # Select at most 5,000 examples
    raw_data = sample_data(
        raw_data,
        columns,
        MAX_SAMPLES
    )

    print(
        f"Using {len(raw_data)} examples "
        f"for fine-tuning."
    )

    # Convert raw examples to training format
    formatted = []

    for example in raw_data:
        try:
            formatted.append(
                format_example(
                    example,
                    columns
                )
            )
        except Exception as e:
            print(
                f"Skipping invalid example: {e}"
            )

    if not formatted:
        raise ValueError(
            "No valid examples were produced."
        )

    print(
        f"Successfully formatted "
        f"{len(formatted)} examples."
    )

    # Split into train / validation / test
    splits = split_data(
        formatted,
        SPLIT_RATIOS
    )

    # Save the splits
    for split_name, split_examples in splits.items():
        save_split(
            split_examples,
            f"{split_name}.jsonl"
        )

    print("Dataset preparation complete.")

        
if __name__ == "__main__":
    main()