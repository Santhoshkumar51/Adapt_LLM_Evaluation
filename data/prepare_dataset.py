"""
Reads raw domain Q&A data, automatically detects common column names,
cleans clearly invalid/duplicate records, formats the data into an
instruction-response prompt template, splits into train/val/test,
and saves as .jsonl files.

Cleaning is intentionally conservative:
- Removes records with missing/empty question or answer.
- Removes malformed records that cannot be represented correctly.
- Removes exact duplicate question-answer pairs after normalization.
- Does NOT remove records based on subjective quality, grammar,
  unusual wording, or domain-specific content.
"""

import json
import random
import os
import csv
import math
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

# Values that clearly indicate a missing field.
MISSING_VALUES = {
    "",
    "null",
    "none",
    "nan",
    "n/a",
    "na"
}


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


def is_missing_value(value):
    """
    Check whether a value clearly represents missing data.

    This is intentionally conservative. Only obvious missing-value
    representations are rejected.
    """

    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    text = str(value).strip().lower()

    return text in MISSING_VALUES


def clean_required_text(value):
    """
    Convert a required field to clean text.

    Returns None when the value is clearly missing.
    """

    if is_missing_value(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def clean_data(data, columns):
    """
    Remove only clearly invalid or duplicate records.

    Invalid records:
        - Missing/empty question
        - Missing/empty answer
        - Malformed row structure

    Duplicate records:
        - Same question + answer after whitespace/case normalization

    Records are NOT removed merely because their content looks unusual.
    """

    cleaned = []

    invalid_count = 0
    duplicate_count = 0

    seen_pairs = set()

    question_col = columns["question"]
    answer_col = columns["answer"]

    for index, example in enumerate(data, start=1):

        # ----------------------------------------------------------
        # Basic row validation
        # ----------------------------------------------------------
        if not isinstance(example, dict):
            invalid_count += 1

            print(
                f"Skipping invalid row {index}: "
                f"row is not a dictionary."
            )

            continue

        # CSV files with malformed rows can sometimes produce
        # unexpected None keys.
        if None in example:
            invalid_count += 1

            print(
                f"Skipping invalid row {index}: "
                f"malformed CSV row."
            )

            continue

        if (
            question_col not in example
            or answer_col not in example
        ):
            invalid_count += 1

            print(
                f"Skipping invalid row {index}: "
                f"required columns are missing."
            )

            continue

        # ----------------------------------------------------------
        # Required question/answer validation
        # ----------------------------------------------------------
        question = clean_required_text(
            example.get(question_col)
        )

        answer = clean_required_text(
            example.get(answer_col)
        )

        if question is None:
            invalid_count += 1

            print(
                f"Skipping invalid row {index}: "
                f"missing/empty question."
            )

            continue

        if answer is None:
            invalid_count += 1

            print(
                f"Skipping invalid row {index}: "
                f"missing/empty answer."
            )

            continue

        # ----------------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------------
        duplicate_key = (
            " ".join(question.split()).lower(),
            " ".join(answer.split()).lower()
        )

        if duplicate_key in seen_pairs:
            duplicate_count += 1

            continue

        seen_pairs.add(duplicate_key)

        # ----------------------------------------------------------
        # Keep original row and normalize required fields
        # ----------------------------------------------------------
        cleaned_example = dict(example)

        cleaned_example[question_col] = question
        cleaned_example[answer_col] = answer

        cleaned.append(cleaned_example)

    print("\nData cleaning summary:")
    print(f"  Original records       : {len(data)}")
    print(f"  Invalid records removed: {invalid_count}")
    print(f"  Duplicate records removed: {duplicate_count}")
    print(f"  Valid unique records   : {len(cleaned)}")

    if not cleaned:
        raise ValueError(
            "No valid records remain after data cleaning."
        )

    return cleaned


def format_example(example: dict, columns: dict) -> dict:
    """Convert one raw example into the canonical training format."""

    question = clean_required_text(
        example.get(columns["question"])
    )

    answer = clean_required_text(
        example.get(columns["answer"])
    )

    if question is None:
        raise ValueError("Question is empty.")

    if answer is None:
        raise ValueError("Answer is empty.")

    context = None

    if columns["context"]:
        value = example.get(columns["context"])

        if not is_missing_value(value):
            context = str(value).strip()

            if not context:
                context = None

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

        if not is_missing_value(category):
            category_text = str(category).strip()

            if category_text:
                formatted["category"] = category_text

    return formatted


def load_raw_data():
    """Load JSONL, JSON, or CSV dataset."""

    extension = os.path.splitext(RAW_FILE)[1].lower()

    if extension == ".jsonl":
        with open(
            RAW_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = []

            for line_number, line in enumerate(f, start=1):

                if not line.strip():
                    continue

                try:
                    data.append(
                        json.loads(line)
                    )
                except json.JSONDecodeError as e:
                    print(
                        f"Skipping malformed JSONL line "
                        f"{line_number}: {e}"
                    )

            return data

    elif extension == ".json":

        with open(
            RAW_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    elif extension == ".csv":

        with open(
            RAW_FILE,
            "r",
            encoding="utf-8",
            newline=""
        ) as f:

            return list(
                csv.DictReader(f)
            )

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

    # No category → simple reproducible random sampling.
    if not category_col:
        rng = random.Random(RANDOM_SEED)

        return rng.sample(
            data,
            max_samples
        )

    # Group examples by category.
    groups = {}

    for example in data:

        category = str(
            example.get(
                category_col,
                "unknown"
            )
        ).strip()

        if not category:
            category = "unknown"

        groups.setdefault(
            category,
            []
        ).append(example)

    total = len(data)

    # --------------------------------------------------------------
    # Proportional allocation using largest-remainder method.
    # This guarantees exactly max_samples when enough records exist.
    # --------------------------------------------------------------
    allocations = {}
    remainders = []

    allocated = 0

    for category, examples in groups.items():

        exact_allocation = (
            len(examples) / total
        ) * max_samples

        base_allocation = int(
            exact_allocation
        )

        # Never allocate more than the group contains.
        base_allocation = min(
            base_allocation,
            len(examples)
        )

        allocations[category] = base_allocation
        allocated += base_allocation

        remainders.append(
            (
                exact_allocation - base_allocation,
                category
            )
        )

    remaining = max_samples - allocated

    # Give remaining slots according to largest fractional remainder.
    remainders.sort(
        key=lambda item: item[0],
        reverse=True
    )

    for _, category in remainders:

        if remaining <= 0:
            break

        if allocations[category] < len(
            groups[category]
        ):
            allocations[category] += 1
            remaining -= 1

    # If any slots remain due to group-size constraints,
    # fill them from categories that still have unused records.
    if remaining > 0:

        for category, examples in groups.items():

            available = (
                len(examples)
                - allocations[category]
            )

            if available <= 0:
                continue

            extra = min(
                available,
                remaining
            )

            allocations[category] += extra
            remaining -= extra

            if remaining == 0:
                break

    # --------------------------------------------------------------
    # Sample records.
    # --------------------------------------------------------------
    rng = random.Random(RANDOM_SEED)

    sampled = []

    for category, examples in groups.items():

        count = min(
            allocations[category],
            len(examples)
        )

        if count > 0:
            sampled.extend(
                rng.sample(
                    examples,
                    count
                )
            )

    rng.shuffle(sampled)

    return sampled


def split_data(data: list, ratios: dict) -> dict:
    """
    Reproducibly split data into train/validation/test sets.
    """

    if not data:
        raise ValueError(
            "Cannot split an empty dataset."
        )

    data = list(data)

    rng = random.Random(RANDOM_SEED)

    rng.shuffle(data)

    n = len(data)

    train_end = int(
        n * ratios["train"]
    )

    val_end = (
        train_end
        + int(n * ratios["val"])
    )

    return {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }


def save_split(split: list, filename: str):
    """Save a dataset split as JSONL."""

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

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
        f"Saved {len(split)} examples to "
        f"{output_path}"
    )


def main():
    print("Loading dataset...")

    raw_data = load_raw_data()

    print(
        f"Loaded {len(raw_data)} examples."
    )

    if not raw_data:
        raise ValueError(
            "Dataset is empty."
        )

    column_map = load_column_map()

    columns = detect_columns(
        raw_data,
        column_map
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    # Clean BEFORE sampling.
    #
    # This ensures that MAX_SAMPLES represents usable records,
    # rather than raw records that may later be discarded.
    # --------------------------------------------------------------
    raw_data = clean_data(
        raw_data,
        columns
    )

    # Select at most 5,000 VALID examples.
    raw_data = sample_data(
        raw_data,
        columns,
        MAX_SAMPLES
    )

    print(
        f"\nUsing {len(raw_data)} valid examples "
        f"for fine-tuning."
    )

    # --------------------------------------------------------------
    # Convert raw examples to training format.
    # --------------------------------------------------------------
    formatted = []

    formatting_errors = 0

    for index, example in enumerate(
        raw_data,
        start=1
    ):

        try:

            formatted.append(
                format_example(
                    example,
                    columns
                )
            )

        except Exception as e:

            formatting_errors += 1

            print(
                f"Skipping invalid example "
                f"{index}: {e}"
            )

    if formatting_errors:
        print(
            f"Formatting errors skipped: "
            f"{formatting_errors}"
        )

    if not formatted:
        raise ValueError(
            "No valid examples were produced."
        )

    print(
        f"Successfully formatted "
        f"{len(formatted)} examples."
    )

    # --------------------------------------------------------------
    # Split into train / validation / test.
    # --------------------------------------------------------------
    splits = split_data(
        formatted,
        SPLIT_RATIOS
    )

    print("\nDataset split:")
    print(
        f"  Train: {len(splits['train'])}"
    )
    print(
        f"  Validation: {len(splits['val'])}"
    )
    print(
        f"  Test: {len(splits['test'])}"
    )

    # --------------------------------------------------------------
    # Save the splits.
    # --------------------------------------------------------------
    for split_name, split_examples in splits.items():

        save_split(
            split_examples,
            f"{split_name}.jsonl"
        )

    print(
        "\nDataset preparation complete."
    )


if __name__ == "__main__":
    main()