"""
Evaluates both the baseline (untouched) and fine-tuned (LoRA-adapted) model
on the held-out test split. Computes accuracy, F1-score, and perplexity.
Saves all metrics and sample predictions to the outputs/ directory.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

OUTPUT_DIR = "outputs/"
METRICS_FILE = "outputs/eval_metrics.json"
PREDICTIONS_FILE = "outputs/sample_predictions.jsonl"
MAX_NEW_TOKENS = 128
NUM_SAMPLE_OUTPUTS = 5


def _generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Generates a response from the model given an input prompt.
    Runs in inference mode with no gradient computation.

    Args:
        model:          Model to generate from (baseline or fine-tuned).
        tokenizer:      Tokenizer matching the model.
        prompt:         Input text prompt.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        Decoded generated text, with prompt stripped from the output.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,         # greedy decoding for deterministic eval
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the input prompt tokens from generated output
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
) -> float:
    """
    Computes mean perplexity over a list of texts.
    Perplexity measures how confidently the model predicts the next token —
    lower is better; a well-adapted model should score lower than baseline.

    Args:
        model:     Model to evaluate.
        tokenizer: Tokenizer matching the model.
        texts:     List of full instruction-response strings.

    Returns:
        Mean perplexity across all texts (float).
    """
    model.eval()
    perplexities = []

    for text in texts:
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            perplexities.append(torch.exp(loss).item())

    return float(np.mean(perplexities))


def _compute_classification_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Computes accuracy and macro F1-score for classification-style tasks
    where the model generates a short category label as output.

    Args:
        predictions: List of model-generated output strings.
        references:  List of ground-truth reference strings.

    Returns:
        Dict containing accuracy and f1 scores.
    """
    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="macro", zero_division=0)
    return {"accuracy": round(accuracy, 4), "f1": round(f1, 4)}


def _collect_sample_outputs(
    baseline_model: PreTrainedModel,
    finetuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_texts: List[str],
    n: int = NUM_SAMPLE_OUTPUTS,
) -> List[Dict[str, str]]:
    """
    Generates n side-by-side sample predictions from both models
    on the same inputs for qualitative dashboard comparison.

    Args:
        baseline_model:  Untouched base model.
        finetuned_model: LoRA fine-tuned model.
        tokenizer:       Shared tokenizer.
        test_texts:      Full instruction-response strings from test set.
        n:               Number of samples to collect.

    Returns:
        List of dicts with keys: input, baseline_output, finetuned_output, reference.
    """
    samples = []
    for text in test_texts[:n]:
        # Extract just the instruction portion as the input prompt
        delimiter = "### Response:"
        if delimiter in text:
            prompt = text[:text.index(delimiter) + len(delimiter)]
            reference = text[text.index(delimiter) + len(delimiter):].strip()
        else:
            prompt = text
            reference = ""

        baseline_out = _generate_response(baseline_model, tokenizer, prompt)
        finetuned_out = _generate_response(finetuned_model, tokenizer, prompt)

        samples.append({
            "input": prompt,
            "baseline_output": baseline_out,
            "finetuned_output": finetuned_out,
            "reference": reference,
        })

    return samples


def run_evaluation(
    baseline_model: PreTrainedModel,
    finetuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
) -> Dict[str, Any]:
    """
    Runs full evaluation of baseline vs fine-tuned model on the test split.
    Computes accuracy, F1, and perplexity for both models.
    Saves metrics to eval_metrics.json and sample outputs to sample_predictions.jsonl.

    Args:
        baseline_model:  Untouched base model.
        finetuned_model: Fine-tuned PEFT model.
        tokenizer:       Shared tokenizer.
        test_dataset:    Tokenized test split.

    Returns:
        Dict containing all computed metrics for both models.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_texts = test_dataset["text"] if "text" in test_dataset.column_names else []

    logger.info("Generating predictions for baseline model...")
    baseline_preds = [
        _generate_response(baseline_model, tokenizer, text)
        for text in test_texts
    ]

    logger.info("Generating predictions for fine-tuned model...")
    finetuned_preds = [
        _generate_response(finetuned_model, tokenizer, text)
        for text in test_texts
    ]

    references = [
        text[text.index("### Response:") + len("### Response:"):].strip()
        if "### Response:" in text else ""
        for text in test_texts
    ]

    logger.info("Computing perplexity...")
    baseline_perplexity = _compute_perplexity(baseline_model, tokenizer, test_texts)
    finetuned_perplexity = _compute_perplexity(finetuned_model, tokenizer, test_texts)

    logger.info("Computing classification metrics...")
    baseline_cls = _compute_classification_metrics(baseline_preds, references)
    finetuned_cls = _compute_classification_metrics(finetuned_preds, references)

    metrics = {
        "baseline": {
            "accuracy": baseline_cls["accuracy"],
            "f1": baseline_cls["f1"],
            "perplexity": round(baseline_perplexity, 4),
        },
        "finetuned": {
            "accuracy": finetuned_cls["accuracy"],
            "f1": finetuned_cls["f1"],
            "perplexity": round(finetuned_perplexity, 4),
        },
        "improvement": {
            "accuracy_delta": round(finetuned_cls["accuracy"] - baseline_cls["accuracy"], 4),
            "f1_delta": round(finetuned_cls["f1"] - baseline_cls["f1"], 4),
            "perplexity_delta": round(baseline_perplexity - finetuned_perplexity, 4),
        },
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", METRICS_FILE)

    logger.info("Collecting sample outputs...")
    samples = _collect_sample_outputs(
        baseline_model, finetuned_model, tokenizer, test_texts
    )

    with open(PREDICTIONS_FILE, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    logger.info("Sample predictions saved to %s", PREDICTIONS_FILE)

    logger.info(
        "Evaluation complete.\n"
        "Baseline   — Accuracy: %.4f | F1: %.4f | Perplexity: %.4f\n"
        "Fine-tuned — Accuracy: %.4f | F1: %.4f | Perplexity: %.4f",
        baseline_cls["accuracy"], baseline_cls["f1"], baseline_perplexity,
        finetuned_cls["accuracy"], finetuned_cls["f1"], finetuned_perplexity,
    )

    return metrics