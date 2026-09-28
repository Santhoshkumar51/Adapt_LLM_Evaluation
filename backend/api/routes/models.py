"""
GET /api/models

Returns the list of base LLMs supported by AdaptEval.
The user selects one of these models from the frontend
before starting a fine-tuning job.
"""

import logging

from fastapi import APIRouter

from backend.api.schemas import (
    ModelInfo,
    ModelsResponse,
    SupportedModel,
)


logger = logging.getLogger(__name__)

router = APIRouter()


SUPPORTED_MODELS: list[ModelInfo] = [
    ModelInfo(
        model_id=SupportedModel.MISTRAL_7B,
        display_name="Mistral 7B",
        parameters="7B",
        description=(
            "Mistral 7B base model for "
            "parameter-efficient fine-tuning."
        ),
    ),

    ModelInfo(
        model_id=SupportedModel.LLAMA3_8B,
        display_name="Llama 3 8B",
        parameters="8B",
        description=(
            "Meta Llama 3 8B base model for "
            "parameter-efficient fine-tuning."
        ),
    ),

    ModelInfo(
        model_id=SupportedModel.QWEN2_7B,
        display_name="Qwen 2 7B",
        parameters="7B",
        description=(
            "Qwen 2 7B base model for "
            "parameter-efficient fine-tuning."
        ),
    ),
]


@router.get(
    "",
    response_model=ModelsResponse,
    summary="List supported base models",
)
async def list_models() -> ModelsResponse:
    """
    Return all base LLMs available for fine-tuning.

    The frontend fetches this endpoint dynamically and
    displays the returned models in the model selector.
    """

    logger.info(
        "Returning list of %d supported models.",
        len(SUPPORTED_MODELS),
    )

    return ModelsResponse(
        models=SUPPORTED_MODELS
    )