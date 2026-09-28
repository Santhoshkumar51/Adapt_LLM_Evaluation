"""
Result and status endpoints for AdaptEval jobs.

GET /api/results/{job_id}/status
GET /api/results/{job_id}
GET /api/results/{job_id}/download
"""

import logging
import os

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from backend.api.schemas import (
    AdapterInfo,
    DatasetStats,
    ImprovementMetrics,
    JobStatus,
    LossPoint,
    ModelMetrics,
    ResultsResponse,
    SamplePrediction,
    StatusResponse,
    TrainingProgress,
)
from backend.store import JOB_STORE


logger = logging.getLogger(__name__)

router = APIRouter()

ADAPTER_BASE_DIR = "adapters"


def _get_job_or_404(job_id: str) -> dict:
    """Return a job or raise HTTP 404."""

    job = JOB_STORE.get(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Job '{job_id}' not found. "
                "Check the job_id returned at submission."
            ),
        )

    return job


@router.get(
    "/{job_id}/status",
    response_model=StatusResponse,
    summary="Poll fine-tuning job status",
)
async def get_job_status(
    job_id: str,
) -> StatusResponse:
    """Return the current lifecycle state and progress."""

    job = _get_job_or_404(job_id)

    progress = None

    if job.get("progress"):
        progress = TrainingProgress(
            **job["progress"]
        )

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
async def get_results(
    job_id: str,
) -> ResultsResponse:
    """Return the complete dashboard payload."""

    job = _get_job_or_404(job_id)

    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job '{job_id}' is not complete yet. "
                f"Current status: {job['status']}."
            ),
        )

    results = job.get("results")

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation results are not available.",
        )

    return ResultsResponse(
        job_id=job_id,
        model_id=job["model_id"],

        baseline=ModelMetrics(
            **results["baseline"]
        ),

        finetuned=ModelMetrics(
            **results["finetuned"]
        ),

        improvement=ImprovementMetrics(
            **results["improvement"]
        ),

        samples=[
            SamplePrediction(**sample)
            for sample in results.get(
                "samples",
                [],
            )
        ],

        adapter=AdapterInfo(
            **results["adapter"]
        ),

        loss_history=[
            LossPoint(**point)
            for point in results.get(
                "loss_history",
                [],
            )
        ],

        dataset_stats=DatasetStats(
            **results.get(
                "dataset_stats",
                {
                    "train": 0,
                    "val": 0,
                    "test": 0,
                    "total": 0,
                },
            )
        ),
    )


@router.get(
    "/{job_id}/download",
    summary="Download LoRA adapter weights",
)
async def download_adapter(
    job_id: str,
) -> FileResponse:
    """
    Download the separately saved LoRA adapter.

    The adapter is not a merged standalone base model.
    """

    job = _get_job_or_404(job_id)

    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Adapter not ready. "
                f"Job status: {job['status']}."
            ),
        )

    adapter_path = os.path.join(
        ADAPTER_BASE_DIR,
        f"adapter_{job_id}",
        "adapter_model.safetensors",
    )

    if not os.path.exists(adapter_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Adapter file not found on disk. "
                "The adapter may not have been saved."
            ),
        )

    logger.info(
        "Serving adapter for job_id=%s from %s",
        job_id,
        adapter_path,
    )

    return FileResponse(
        path=adapter_path,
        filename=(
            f"adapteval_adapter_"
            f"{job_id[:8]}.safetensors"
        ),
        media_type="application/octet-stream",
    )