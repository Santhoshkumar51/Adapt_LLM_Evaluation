"""
POST /api/finetune
Accepts a model selection and a .jsonl training data file upload.
Validates the file, persists it to disk, and kicks off the full
AdaptEval fine-tuning pipeline as a background task.
Returns a job_id immediately — the frontend polls /api/results/{job_id}/status.
"""

import logging
import os
import uuid
from typing import Dict, Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from backend.api.schemas import (
    FinetuneJobResponse,
    FinetuneRequest,
    JobStatus,
    SupportedModel,
)
from backend.pipeline import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024   # 20 MB hard cap on uploaded training data
ALLOWED_EXTENSIONS  = {".jsonl", ".csv"}
UPLOAD_DIR          = "data/uploads"

# ── Shared job store (injected from main.py at import time) ──────────────────
# Imported here to allow background task updates to be visible via /status route.
from backend.store import JOB_STORE


def _validate_file(filename: str, file_size: int) -> None:
    """
    Validates the uploaded training data file by extension and size.

    Args:
        filename:  Original filename from the upload.
        file_size: File size in bytes.

    Raises:
        HTTPException 400: If extension is unsupported or file exceeds the size cap.
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Upload a .jsonl or .csv file.",
        )
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size of 20 MB.",
        )


async def _persist_upload(file: UploadFile, job_id: str) -> str:
    """
    Reads the uploaded file and saves it to disk under a job-specific path.

    Args:
        file:   FastAPI UploadFile object.
        job_id: UUID string for this job — used to namespace the saved file.

    Returns:
        Absolute path to the saved file on disk.
    """
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)

    dest_path = os.path.join(job_upload_dir, file.filename)
    contents = await file.read()

    with open(dest_path, "wb") as f:
        f.write(contents)

    logger.info("Uploaded file saved to: %s (%d bytes)", dest_path, len(contents))
    return dest_path


@router.post(
    "",
    response_model=FinetuneJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a fine-tuning job",
)
async def submit_finetune_job(
    background_tasks: BackgroundTasks,
    model_id: str = Form(..., description="Hugging Face model ID selected by the user"),
    file: UploadFile = File(..., description="Training data file (.jsonl or .csv)"),
) -> FinetuneJobResponse:
    """
    Accepts a model selection and training data upload.
    Validates the file, persists it to disk, and starts the fine-tuning
    pipeline as a background task. Returns a job_id for status polling.

    The response is HTTP 202 Accepted — fine-tuning is asynchronous.
    The client should poll GET /api/results/{job_id}/status for progress.
    """
    # Validate model_id against the enum of supported models
    if model_id not in [m.value for m in SupportedModel]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model_id '{model_id}'. Call GET /api/models for valid options.",
        )

    # Read file into memory to check size before writing to disk
    file_contents = await file.read()
    await file.seek(0)   # reset so _persist_upload can re-read

    _validate_file(file.filename, len(file_contents))

    job_id = str(uuid.uuid4())

    # Initialise job state before background task starts
    JOB_STORE[job_id] = {
        "status":   JobStatus.QUEUED,
        "model_id": model_id,
        "progress": None,
        "results":  None,
        "error":    None,
    }

    upload_path = await _persist_upload(file, job_id)

    logger.info(
        "Fine-tuning job accepted | job_id=%s | model=%s | file=%s",
        job_id, model_id, upload_path,
    )

    # Kick off the full pipeline in the background
    # FastAPI will return the 202 response immediately
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
        message=f"Fine-tuning job queued. Poll /api/results/{job_id}/status for progress.",
    )