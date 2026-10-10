"""
Evaluates baseline and LoRA fine-tuned models on the held-out test split.

Evaluation is performed one model at a time so that low-VRAM GPUs do not
need to keep both the baseline and fine-tuned models in memory simultaneously.

Metrics:
    ROUGE-L F1
    Perplexity

Outputs:
    outputs/eval_metrics.json
    outputs/sample_predictions.jsonl
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


OUTPUT_DIR = "outputs/"
METRICS_FILE = "outputs/eval_metrics.json"
PREDICTIONS_FILE = "outputs/sample_predictions.jsonl"

MAX_NEW_TOKENS = 128
NUM_SAMPLE_OUTPUTS = 5

RESPONSE_DELIMITER = "### Response:"


# =============================================================================
# Generation
# =============================================================================

def _generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:

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

    generated_ids = output_ids[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


# =============================================================================
# Perplexity
# =============================================================================

def _compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
) -> float:

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

            perplexities.append(
                torch.exp(
                    outputs.loss
                ).item()
            )

    return (
        float(np.mean(perplexities))
        if perplexities
        else 0.0
    )


# =============================================================================
# ROUGE-L
# =============================================================================

def _tokenize_for_rouge(
    text: str,
) -> List[str]:

    return re.findall(
        r"\w+|[^\w\s]",
        text.lower(),
        flags=re.UNICODE,
    )


def _lcs_length(
    a: List[str],
    b: List[str],
) -> int:

    if not a or not b:
        return 0

    short, long_ = (
        (a, b)
        if len(a) <= len(b)
        else (b, a)
    )

    previous = [
        0
    ] * (
        len(short) + 1
    )

    for token in long_:

        current = [
            0
        ]

        for j, short_token in enumerate(
            short,
            start=1,
        ):

            if token == short_token:

                current.append(
                    previous[j - 1] + 1
                )

            else:

                current.append(
                    max(
                        previous[j],
                        current[-1],
                    )
                )

        previous = current

    return previous[-1]


def _rouge_l_f1(
    prediction: str,
    reference: str,
) -> float:

    pred_tokens = _tokenize_for_rouge(
        prediction
    )

    ref_tokens = _tokenize_for_rouge(
        reference
    )

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(
        pred_tokens,
        ref_tokens,
    )

    precision = (
        lcs / len(pred_tokens)
    )

    recall = (
        lcs / len(ref_tokens)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def _compute_rouge_l(
    predictions: List[str],
    references: List[str],
) -> float:

    if not predictions or not references:
        return 0.0

    scores = [
        _rouge_l_f1(
            pred,
            ref,
        )
        for pred, ref in zip(
            predictions,
            references,
        )
    ]

    return float(
        np.mean(scores)
    )


# =============================================================================
# Prompt / reference handling
# =============================================================================

def _split_prompt_and_reference(
    text: str,
):

    if RESPONSE_DELIMITER not in text:

        return (
            text,
            "",
        )

    split_at = text.index(
        RESPONSE_DELIMITER
    )

    prompt = (
        text[
            :split_at
            + len(RESPONSE_DELIMITER)
        ]
    )

    reference = (
        text[
            split_at
            + len(RESPONSE_DELIMITER):
        ].strip()
    )

    return (
        prompt,
        reference,
    )


# =============================================================================
# Single-model evaluation
# =============================================================================

def evaluate_single_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
    model_label: str,
) -> Dict[str, Any]:
    """
    Evaluate exactly one model at a time.

    Returns:
        {
            "rouge_l": float,
            "perplexity": float,
            "predictions": [...],
            "references": [...],
            "prompts": [...]
        }
    """

    if "text" not in test_dataset.column_names:

        raise ValueError(
            "test_dataset must contain a 'text' column."
        )

    test_texts = test_dataset[
        "text"
    ]

    if not test_texts:

        raise ValueError(
            "test_dataset is empty."
        )

    prompts = []
    references = []

    for text in test_texts:

        prompt, reference = (
            _split_prompt_and_reference(
                text
            )
        )

        prompts.append(prompt)
        references.append(reference)

    logger.info(
        "Generating %s model responses (%d examples)...",
        model_label,
        len(prompts),
    )

    predictions = [
        _generate_response(
            model,
            tokenizer,
            prompt,
        )
        for prompt in prompts
    ]

    logger.info(
        "Computing %s ROUGE-L...",
        model_label,
    )

    rouge_l = _compute_rouge_l(
        predictions,
        references,
    )

    logger.info(
        "Computing %s perplexity...",
        model_label,
    )

    perplexity = _compute_perplexity(
        model,
        tokenizer,
        test_texts,
    )

    return {
        "rouge_l": round(
            rouge_l,
            4,
        ),
        "perplexity": round(
            perplexity,
            4,
        ),
        "predictions": predictions,
        "references": references,
        "prompts": prompts,
    }


# =============================================================================
# Sample comparison
# =============================================================================

def build_sample_comparisons(
    baseline_result: Dict[str, Any],
    finetuned_result: Dict[str, Any],
    n: int = NUM_SAMPLE_OUTPUTS,
) -> List[Dict[str, str]]:

    samples = []

    count = min(
        n,
        len(
            baseline_result[
                "predictions"
            ]
        ),
        len(
            finetuned_result[
                "predictions"
            ]
        ),
    )

    for i in range(count):

        samples.append(
            {
                "input": baseline_result[
                    "prompts"
                ][i],

                "baseline_output":
                    baseline_result[
                        "predictions"
                    ][i],

                "finetuned_output":
                    finetuned_result[
                        "predictions"
                    ][i],

                "reference":
                    baseline_result[
                        "references"
                    ][i],
            }
        )

    return samples


# =============================================================================
# Save evaluation outputs
# =============================================================================

def save_evaluation_results(
    baseline_result: Dict[str, Any],
    finetuned_result: Dict[str, Any],
) -> Dict[str, Any]:

    baseline_rouge = baseline_result[
        "rouge_l"
    ]

    finetuned_rouge = finetuned_result[
        "rouge_l"
    ]

    baseline_ppl = baseline_result[
        "perplexity"
    ]

    finetuned_ppl = finetuned_result[
        "perplexity"
    ]

    metrics = {

        "baseline": {
            "rouge_l": baseline_rouge,
            "perplexity": baseline_ppl,
        },

        "finetuned": {
            "rouge_l": finetuned_rouge,
            "perplexity": finetuned_ppl,
        },

        "improvement": {
            "rouge_l_delta": round(
                finetuned_rouge
                - baseline_rouge,
                4,
            ),

            "perplexity_delta": round(
                baseline_ppl
                - finetuned_ppl,
                4,
            ),
        },
    }

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    logger.info(
        "Metrics saved → %s",
        METRICS_FILE,
    )

    samples = build_sample_comparisons(
        baseline_result,
        finetuned_result,
    )

    with open(
        PREDICTIONS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for sample in samples:

            f.write(
                json.dumps(sample)
                + "\n"
            )

    logger.info(
        "Sample predictions saved → %s",
        PREDICTIONS_FILE,
    )

    logger.info(
        "Evaluation complete — "
        "Baseline: ROUGE-L=%.4f | Perplexity=%.2f | "
        "Fine-tuned: ROUGE-L=%.4f | Perplexity=%.2f",
        baseline_rouge,
        baseline_ppl,
        finetuned_rouge,
        finetuned_ppl,
    )

    return metrics