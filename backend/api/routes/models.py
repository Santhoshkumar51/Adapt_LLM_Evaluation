"""
GET /api/models
Returns the list of base LLMs supported by AdaptEval.
The user picks from this list in the frontend dropdown — no model upload needed.
"""

import logging
from fastapi import APIRouter
from backend.api.schemas import ModelInfo, ModelsResponse, SupportedModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Static catalogue of supported models ─────────────────────────────────────
# Only models verified to fit on a single T4 GPU with 4-bit QLoRA are listed.
SUPPORTED_MODELS: list[ModelInfo] = [
    ModelInfo(
        model_id=SupportedModel.MISTRAL_7B,
        display_name="Mistral 7B",
        parameters="7B",
        description="Strong general-purpose instruction model. Recommended default.",
    ),
    ModelInfo(
        model_id=SupportedModel.LLAMA3_8B,
        display_name="Llama 3 8B",
        parameters="8B",
        description="Meta's Llama 3 base model. Excellent for English-language tasks.",
    ),
    ModelInfo(
        model_id=SupportedModel.QWEN2_7B,
        display_name="Qwen 2 7B",
        parameters="7B",
        description="Alibaba's Qwen 2. Strong on structured output and multilingual tasks.",
    ),
]


@router.get("", response_model=ModelsResponse, summary="List supported base models")
async def list_models() -> ModelsResponse:
    """
    Returns all base LLMs available for fine-tuning in AdaptEval.
    The frontend renders this as a dropdown — the user selects one model per run.
    """
    logger.info("Returning list of %d supported models.", len(SUPPORTED_MODELS))
    return ModelsResponse(models=SUPPORTED_MODELS)