"""
GET /api/results/{job_id}/status  — poll training progress
GET /api/results/{job_id}         — fetch full evaluation results
GET /api/results/{job_id}/download — stream the adapter .safetensors file
"""

import logging
import os

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from backend.api.schemas import (
    AdapterInfo,
    ImprovementMetrics,
    JobStatus,
    ModelMetrics,
    ResultsResponse,
    SamplePrediction,
    StatusResponse,
    TrainingProgress,
)
from backend.store import JOB_STORE

logger = logging.getLogger(__name__)
router = APIRouter()

ADAPTER_BASE_DIR = "adapters/"


def _get_job_or_404(job_id: str) -> dict:
    """
    Fetches job state from JOB_STORE or raises HTTP 404.

    Args:
        job_id: UUID string for the fine-tuning job.

    Returns:
        Job state dict from JOB_STORE.

    Raises:
        HTTPException 404: If job_id is not found in JOB_STORE.
    """
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found. Check the job_id returned at submission.",
        )
    return job


@router.get(
    "/{job_id}/status",
    response_model=StatusResponse,
    summary="Poll fine-tuning job status",
)
async def get_job_status(job_id: str) -> StatusResponse:
    """
    Returns the current lifecycle state and training progress for a job.
    Frontend polls this every 10–15 seconds to update a progress indicator.

    Possible statuses: queued → preparing → training → evaluating → complete | failed
    """
    job = _get_job_or_404(job_id)

    progress = None
    if job["progress"]:
        progress = TrainingProgress(**job["progress"])

    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=progress,
        error=job.get("error"),
    )


@router.get(
    "/{job_id}",
    response_model=ResultsResponse,
    summary="Fetch full evaluation results",
)
async def get_results(job_id: str) -> ResultsResponse:
    """
    Returns the full evaluation dashboard payload for a completed job.
    Only available when job status is COMPLETE — returns HTTP 409 otherwise.
    Powers the entire AdaptEval results dashboard (KPIs + sample outputs).
    """
    job = _get_job_or_404(job_id)

    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job '{job_id}' is not complete yet. Current status: {job['status']}.",
        )

    results = job["results"]

    return ResultsResponse(
        job_id=job_id,
        model_id=job["model_id"],
        baseline=ModelMetrics(**results["baseline"]),
        finetuned=ModelMetrics(**results["finetuned"]),
        improvement=ImprovementMetrics(**results["improvement"]),
        samples=[SamplePrediction(**s) for s in results["samples"]],
        adapter=AdapterInfo(**results["adapter"]),
    )


@router.get(
    "/{job_id}/download",
    summary="Download adapter weights file",
)
async def download_adapter(job_id: str) -> FileResponse:
    """
    Streams the merged adapter .safetensors file to the client.
    Only available when job status is COMPLETE.
    This is the primary take-away artifact of the fine-tuning run.
    """
    job = _get_job_or_404(job_id)

    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Adapter not ready. Job status: {job['status']}.",
        )

    adapter_path = os.path.join(
        ADAPTER_BASE_DIR,
        f"adapter_{job_id}",
        "adapter_model.safetensors",
    )

    if not os.path.exists(adapter_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Adapter file not found on disk. The job may have failed during merge.",
        )

    logger.info("Serving adapter file for job_id=%s from %s", job_id, adapter_path)

    return FileResponse(
        path=adapter_path,
        filename=f"adapteval_adapter_{job_id[:8]}.safetensors",
        media_type="application/octet-stream",
    )