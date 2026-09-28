"""
Evaluates baseline and LoRA fine-tuned models on the held-out test split.

Metrics computed:
    ROUGE-L F1  — measures longest-common-subsequence overlap between
                  generated and reference responses. Appropriate for
                  generative text tasks.
    Perplexity  — measures how confidently the model predicts the next
                  token over full instruction-response sequences.
                  Lower is better; a well-adapted model should score
                  significantly lower than baseline.

Outputs:
    outputs/eval_metrics.json          — all metric values
    outputs/sample_predictions.jsonl   — 5 side-by-side sample outputs
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

OUTPUT_DIR       = "outputs/"
METRICS_FILE     = "outputs/eval_metrics.json"
PREDICTIONS_FILE = "outputs/sample_predictions.jsonl"
MAX_NEW_TOKENS   = 128
NUM_SAMPLE_OUTPUTS = 5
RESPONSE_DELIMITER = "### Response:"


# ── Generation ────────────────────────────────────────────────────────────────

def _generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Generate a response from a model given an input prompt.
    Runs greedy decoding (do_sample=False) for deterministic,
    reproducible evaluation outputs.

    Args:
        model:          Model to generate from.
        tokenizer:      Matching tokenizer.
        prompt:         Full instruction prompt string.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Decoded generated text with prompt tokens stripped.
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
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip input tokens — keep only newly generated tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ── Perplexity ────────────────────────────────────────────────────────────────

def _compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
) -> float:
    """
    Compute mean perplexity over full instruction-response sequences.
    Perplexity = exp(cross-entropy loss) — lower means the model assigns
    higher probability to the correct next token at each position.

    Args:
        model:     Model to evaluate.
        tokenizer: Matching tokenizer.
        texts:     List of full formatted instruction-response strings.

    Returns:
        Mean perplexity across all texts (float). Returns 0.0 if empty.
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
            outputs = model(
                **encodings,
                labels=encodings["input_ids"],
            )
            perplexities.append(torch.exp(outputs.loss).item())

    return float(np.mean(perplexities)) if perplexities else 0.0


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def _tokenize_for_rouge(text: str) -> List[str]:
    """
    Tokenize text into word and punctuation tokens for ROUGE-L computation.
    Lowercases for case-insensitive matching.

    Args:
        text: Input string.

    Returns:
        List of lowercase word/punctuation tokens.
    """
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """
    Compute the length of the Longest Common Subsequence (LCS) of two
    token lists using a space-optimised O(min(m,n)) DP algorithm.

    Args:
        a: First token list.
        b: Second token list.

    Returns:
        LCS length (int).
    """
    if not a or not b:
        return 0

    # Always iterate over the longer list, keep the shorter in the array
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    previous = [0] * (len(short) + 1)

    for token in long_:
        current = [0]
        for j, short_token in enumerate(short, start=1):
            if token == short_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current

    return previous[-1]


def _rouge_l_f1(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 for a single prediction/reference pair.
    F1 = 2 * precision * recall / (precision + recall)
    where precision = LCS/len(prediction) and recall = LCS/len(reference).

    Args:
        prediction: Model-generated text.
        reference:  Ground-truth reference text.

    Returns:
        ROUGE-L F1 score in [0, 1]. Returns 0.0 for empty strings.
    """
    pred_tokens = _tokenize_for_rouge(prediction)
    ref_tokens  = _tokenize_for_rouge(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall    = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _compute_rouge_l(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute mean ROUGE-L F1 across all prediction/reference pairs.

    Args:
        predictions: List of model-generated outputs.
        references:  List of ground-truth reference strings.

    Returns:
        Mean ROUGE-L F1 score. Returns 0.0 if lists are empty.
    """
    if not predictions or not references:
        return 0.0

    scores = [
        _rouge_l_f1(pred, ref)
        for pred, ref in zip(predictions, references)
    ]
    return float(np.mean(scores))


# ── Sample collection ─────────────────────────────────────────────────────────

def _split_prompt_and_reference(text: str):
    """
    Split a formatted instruction-response string into its prompt
    (everything up to and including '### Response:') and the
    reference answer (everything after).

    Args:
        text: Full formatted string.

    Returns:
        Tuple of (prompt, reference). If delimiter not found,
        returns (text, "").
    """
    if RESPONSE_DELIMITER not in text:
        return text, ""

    split_at  = text.index(RESPONSE_DELIMITER)
    prompt    = text[:split_at + len(RESPONSE_DELIMITER)]
    reference = text[split_at + len(RESPONSE_DELIMITER):].strip()
    return prompt, reference


def _collect_sample_outputs(
    baseline_model: PreTrainedModel,
    finetuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_texts: List[str],
    n: int = NUM_SAMPLE_OUTPUTS,
) -> List[Dict[str, str]]:
    """
    Generate n side-by-side output comparisons from both models
    on the same input prompts for qualitative dashboard display.

    Args:
        baseline_model:  Untouched base model.
        finetuned_model: LoRA fine-tuned model.
        tokenizer:       Shared tokenizer.
        test_texts:      Full instruction-response strings from test split.
        n:               Number of samples to collect.

    Returns:
        List of dicts with keys: input, baseline_output,
        finetuned_output, reference.
    """
    samples = []

    for text in test_texts[:n]:
        prompt, reference = _split_prompt_and_reference(text)

        baseline_output  = _generate_response(baseline_model,  tokenizer, prompt)
        finetuned_output = _generate_response(finetuned_model, tokenizer, prompt)

        samples.append({
            "input":            prompt,
            "baseline_output":  baseline_output,
            "finetuned_output": finetuned_output,
            "reference":        reference,
        })

    return samples


# ── Main evaluation entry point ───────────────────────────────────────────────

def run_evaluation(
    baseline_model: PreTrainedModel,
    finetuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
) -> Dict[str, Any]:
    """
    Run full evaluation of baseline vs fine-tuned model on the test split.

    Steps:
        1. Extract prompts and reference answers from test_dataset.
        2. Generate responses from both models on all prompts.
        3. Compute ROUGE-L F1 for both sets of responses.
        4. Compute perplexity for both models over full sequences.
        5. Write eval_metrics.json and sample_predictions.jsonl.

    Args:
        baseline_model:  Untouched base model (frozen weights).
        finetuned_model: LoRA-adapted fine-tuned model.
        tokenizer:       Shared tokenizer for both models.
        test_dataset:    Tokenized test split with a 'text' column.

    Returns:
        Metrics dict with keys: baseline, finetuned, improvement.
        Each contains rouge_l and perplexity values.

    Raises:
        ValueError: If test_dataset has no 'text' column or is empty.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if "text" not in test_dataset.column_names:
        raise ValueError(
            "test_dataset must contain a 'text' column with formatted "
            "instruction-response strings."
        )

    test_texts = test_dataset["text"]

    if not test_texts:
        raise ValueError(
            "test_dataset is empty — no examples to evaluate."
        )

    # ── Extract prompts and references ────────────────────────────────
    prompts    = []
    references = []

    for text in test_texts:
        prompt, reference = _split_prompt_and_reference(text)
        prompts.append(prompt)
        references.append(reference)

    # ── Generate responses ────────────────────────────────────────────
    logger.info("Generating baseline model responses (%d examples)...", len(prompts))
    baseline_preds = [
        _generate_response(baseline_model, tokenizer, prompt)
        for prompt in prompts
    ]

    logger.info("Generating fine-tuned model responses (%d examples)...", len(prompts))
    finetuned_preds = [
        _generate_response(finetuned_model, tokenizer, prompt)
        for prompt in prompts
    ]

    # ── Compute ROUGE-L ───────────────────────────────────────────────
    logger.info("Computing ROUGE-L...")
    baseline_rouge_l  = _compute_rouge_l(baseline_preds,  references)
    finetuned_rouge_l = _compute_rouge_l(finetuned_preds, references)

    # ── Compute perplexity ────────────────────────────────────────────
    logger.info("Computing perplexity...")
    baseline_perplexity  = _compute_perplexity(baseline_model,  tokenizer, test_texts)
    finetuned_perplexity = _compute_perplexity(finetuned_model, tokenizer, test_texts)

    # ── Build metrics dict ────────────────────────────────────────────
    metrics = {
        "baseline": {
            "rouge_l":    round(baseline_rouge_l,    4),
            "perplexity": round(baseline_perplexity, 4),
        },
        "finetuned": {
            "rouge_l":    round(finetuned_rouge_l,    4),
            "perplexity": round(finetuned_perplexity, 4),
        },
        "improvement": {
            "rouge_l_delta":    round(finetuned_rouge_l    - baseline_rouge_l,    4),
            "perplexity_delta": round(baseline_perplexity  - finetuned_perplexity, 4),
        },
    }

    # ── Save metrics ──────────────────────────────────────────────────
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", METRICS_FILE)

    # ── Collect and save sample predictions ──────────────────────────
    logger.info("Collecting %d sample output comparisons...", NUM_SAMPLE_OUTPUTS)
    samples = _collect_sample_outputs(
        baseline_model,
        finetuned_model,
        tokenizer,
        test_texts,
    )

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    logger.info("Sample predictions saved → %s", PREDICTIONS_FILE)

    logger.info(
        "Evaluation complete — "
        "Baseline:  ROUGE-L=%.4f | Perplexity=%.2f | "
        "Fine-tuned: ROUGE-L=%.4f | Perplexity=%.2f | "
        "Δ ROUGE-L=+%.4f | Δ Perplexity=-%.2f",
        baseline_rouge_l,  baseline_perplexity,
        finetuned_rouge_l, finetuned_perplexity,
        metrics["improvement"]["rouge_l_delta"],
        metrics["improvement"]["perplexity_delta"],
    )

    return metrics