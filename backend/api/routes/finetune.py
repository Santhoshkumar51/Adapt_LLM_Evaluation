"""
POST /api/finetune

Accepts a selected base model and a training dataset,
validates the request, stores the uploaded file, and
starts the AdaptEval pipeline as a background task.
"""

import logging
import os
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from backend.api.schemas import (
    FinetuneJobResponse,
    JobStatus,
    SupportedModel,
)
from backend.pipeline import run_pipeline
from backend.store import JOB_STORE


logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".jsonl",
    ".csv",
}

UPLOAD_DIR = "data/uploads"


def _validate_file(
    filename: str,
    file_size: int,
) -> None:
    """Validate uploaded training data."""

    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A training data file is required.",
        )

    extension = os.path.splitext(
        filename
    )[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type "
                f"'{extension}'. "
                "Upload a .jsonl or .csv file."
            ),
        )

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "File exceeds the maximum "
                "allowed size of 50 MB."
            ),
        )


async def _persist_upload(
    file: UploadFile,
    job_id: str,
) -> str:
    """Save an uploaded file in its job-specific directory."""

    job_upload_dir = os.path.join(
        UPLOAD_DIR,
        job_id,
    )

    os.makedirs(
        job_upload_dir,
        exist_ok=True,
    )

    filename = os.path.basename(
        file.filename or "training_data"
    )

    destination = os.path.join(
        job_upload_dir,
        filename,
    )

    contents = await file.read()

    with open(
        destination,
        "wb",
    ) as output_file:
        output_file.write(contents)

    logger.info(
        "Uploaded file saved to %s (%d bytes)",
        destination,
        len(contents),
    )

    return destination


@router.post(
    "",
    response_model=FinetuneJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a fine-tuning job",
)
async def submit_finetune_job(
    background_tasks: BackgroundTasks,
    model_id: str = Form(
        ...,
        description=(
            "Hugging Face model ID selected by "
            "the user"
        ),
    ),
    file: UploadFile = File(
        ...,
        description=(
            "Training data file (.jsonl or .csv)"
        ),
    ),
) -> FinetuneJobResponse:
    """
    Validate the selected model and dataset,
    create a job, and start the pipeline.
    """

    supported_models = {
        model.value
        for model in SupportedModel
    }

    if model_id not in supported_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported model_id '{model_id}'. "
                "Call GET /api/models for valid options."
            ),
        )

    file_contents = await file.read()

    await file.seek(0)

    _validate_file(
        file.filename or "",
        len(file_contents),
    )

    job_id = str(
        uuid.uuid4()
    )

    JOB_STORE[job_id] = {
        "status": JobStatus.QUEUED,
        "model_id": model_id,
        "progress": None,
        "results": None,
        "error": None,
    }

    upload_path = await _persist_upload(
        file,
        job_id,
    )

    logger.info(
        "Fine-tuning job accepted | "
        "job_id=%s | model=%s | file=%s",
        job_id,
        model_id,
        upload_path,
    )

    background_tasks.add_task(
        run_pipeline,
        job_id=job_id,
        model_id=model_id,
        data_file_path=upload_path,
        job_store=JOB_STORE,
    )

    return FinetuneJobResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message=(
            "Fine-tuning job queued. "
            f"Poll /api/results/{job_id}/status "
            "for progress."
        ),
    )