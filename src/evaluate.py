"""
Evaluates the baseline and LoRA fine-tuned model on the held-out test split.
Computes ROUGE-L and perplexity and stores representative sample outputs.
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

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()


def _compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
) -> float:
    """
    Compute mean perplexity over full instruction-response sequences.
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
                labels=encodings["input_ids"]
            )

            perplexities.append(
                torch.exp(outputs.loss).item()
            )

    return (
        float(np.mean(perplexities))
        if perplexities
        else 0.0
    )


def _tokenize_for_rouge(text: str) -> List[str]:
    """
    Tokenize text into words and punctuation for ROUGE-L.
    """

    return re.findall(
        r"\w+|[^\w\s]",
        text.lower(),
        flags=re.UNICODE,
    )


def _lcs_length(
    a: List[str],
    b: List[str],
) -> int:
    """
    Compute longest common subsequence length.
    """

    if not a or not b:
        return 0

    if len(a) < len(b):
        short = a
        long_ = b
    else:
        short = b
        long_ = a

    previous = [0] * (len(short) + 1)

    for token in long_:

        current = [0]

        for j, short_token in enumerate(
            short,
            start=1
        ):

            if token == short_token:

                current.append(
                    previous[j - 1] + 1
                )

            else:

                current.append(
                    max(
                        previous[j],
                        current[-1]
                    )
                )

        previous = current

    return previous[-1]


def _rouge_l_f1(
    prediction: str,
    reference: str,
) -> float:
    """
    Compute ROUGE-L F1 for one prediction/reference pair.
    """

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
        ref_tokens
    )

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def _compute_rouge_l(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute mean ROUGE-L F1 across the test set.
    """

    scores = [
        _rouge_l_f1(
            prediction,
            reference
        )
        for prediction, reference
        in zip(predictions, references)
    ]

    return (
        float(np.mean(scores))
        if scores
        else 0.0
    )


def _collect_sample_outputs(
    baseline_model: PreTrainedModel,
    finetuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_texts: List[str],
    n: int = NUM_SAMPLE_OUTPUTS,
) -> List[Dict[str, str]]:
    """
    Generate representative side-by-side outputs.
    """

    samples = []

    for text in test_texts[:n]:

        delimiter = "### Response:"

        if delimiter in text:

            split_at = text.index(
                delimiter
            )

            prompt = (
                text[
                    :split_at
                    + len(delimiter)
                ]
            )

            reference = (
                text[
                    split_at
                    + len(delimiter):
                ].strip()
            )

        else:

            prompt = text
            reference = ""

        baseline_output = _generate_response(
            baseline_model,
            tokenizer,
            prompt,
        )

        finetuned_output = _generate_response(
            finetuned_model,
            tokenizer,
            prompt,
        )

        samples.append({
            "input": prompt,
            "baseline_output": baseline_output,
            "finetuned_output": finetuned_output,
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
    Evaluate baseline vs fine-tuned models using ROUGE-L and perplexity.

    Perplexity is computed on the full formatted
    instruction-response sequences.

    ROUGE-L is computed between generated responses
    and reference responses.
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    test_texts = (
        test_dataset["text"]
        if "text" in test_dataset.column_names
        else []
    )

    if not test_texts:
        raise ValueError(
            "The test dataset does not contain "
            "any evaluation texts."
        )

    prompts = []
    references = []

    for text in test_texts:

        delimiter = "### Response:"

        if delimiter in text:

            split_at = text.index(
                delimiter
            )

            prompts.append(
                text[
                    :split_at
                    + len(delimiter)
                ]
            )

            references.append(
                text[
                    split_at
                    + len(delimiter):
                ].strip()
            )

        else:

            prompts.append(text)
            references.append("")

    logger.info(
        "Generating baseline responses..."
    )

    baseline_preds = [
        _generate_response(
            baseline_model,
            tokenizer,
            prompt
        )
        for prompt in prompts
    ]

    logger.info(
        "Generating fine-tuned responses..."
    )

    finetuned_preds = [
        _generate_response(
            finetuned_model,
            tokenizer,
            prompt
        )
        for prompt in prompts
    ]

    logger.info(
        "Computing ROUGE-L..."
    )

    baseline_rouge_l = _compute_rouge_l(
        baseline_preds,
        references
    )

    finetuned_rouge_l = _compute_rouge_l(
        finetuned_preds,
        references
    )

    logger.info(
        "Computing perplexity..."
    )

    baseline_perplexity = _compute_perplexity(
        baseline_model,
        tokenizer,
        test_texts
    )

    finetuned_perplexity = _compute_perplexity(
        finetuned_model,
        tokenizer,
        test_texts
    )

    metrics = {

        "baseline": {
            "rouge_l": round(
                baseline_rouge_l,
                4
            ),
            "perplexity": round(
                baseline_perplexity,
                4
            ),
        },

        "finetuned": {
            "rouge_l": round(
                finetuned_rouge_l,
                4
            ),
            "perplexity": round(
                finetuned_perplexity,
                4
            ),
        },

        "improvement": {
            "rouge_l_delta": round(
                finetuned_rouge_l
                - baseline_rouge_l,
                4
            ),
            "perplexity_delta": round(
                baseline_perplexity
                - finetuned_perplexity,
                4
            ),
        },
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    logger.info(
        "Collecting sample outputs..."
    )

    samples = _collect_sample_outputs(
        baseline_model,
        finetuned_model,
        tokenizer,
        test_texts,
    )

    with open(
        PREDICTIONS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for sample in samples:

            f.write(
                json.dumps(sample)
                + "\n"
            )

    logger.info(
        "Evaluation complete | "
        "baseline ROUGE-L %.4f, perplexity %.4f | "
        "fine-tuned ROUGE-L %.4f, perplexity %.4f",
        baseline_rouge_l,
        baseline_perplexity,
        finetuned_rouge_l,
        finetuned_perplexity,
    )

    return metrics