"""
Pydantic schemas for all AdaptEval API request and response payloads.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SupportedModel(str, Enum):
    MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
    QWEN2_7B   = "Qwen/Qwen2-7B"
    TINYLLAMA  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class ModelInfo(BaseModel):
    model_id:     str = Field(..., description="Hugging Face model ID")
    display_name: str = Field(..., description="Human-readable name for UI dropdown")
    parameters:   str = Field(..., description="Approximate parameter count")
    description:  str = Field(..., description="One-line model description")


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class JobStatus(str, Enum):
    QUEUED     = "queued"
    PREPARING  = "preparing"
    TRAINING   = "training"
    EVALUATING = "evaluating"
    COMPLETE   = "complete"
    FAILED     = "failed"


class FinetuneRequest(BaseModel):
    model_id: SupportedModel = Field(..., description="Selected base model ID")


class FinetuneJobResponse(BaseModel):
    job_id:  str       = Field(..., description="UUID for this fine-tuning run")
    status:  JobStatus = Field(..., description="Initial job status")
    message: str       = Field(..., description="Human-readable status message")


class TrainingProgress(BaseModel):
    current_epoch: int   = Field(..., description="Current epoch")
    total_epochs:  int   = Field(..., description="Total epochs configured")
    train_loss:    float = Field(..., description="Latest training loss")
    eval_loss:     float = Field(..., description="Latest validation loss")
    elapsed_mins:  float = Field(..., description="Elapsed time in minutes")


class StatusResponse(BaseModel):
    job_id:   str                       = Field(..., description="Job UUID")
    status:   JobStatus                 = Field(..., description="Current lifecycle state")
    progress: Optional[TrainingProgress] = Field(None, description="Training progress")
    error:    Optional[str]             = Field(None, description="Error message if failed")


class ModelMetrics(BaseModel):
    rouge_l:    float = Field(0.0, description="ROUGE-L F1 score on held-out test set")
    perplexity: float = Field(0.0, description="Mean perplexity on held-out test set")


class ImprovementMetrics(BaseModel):
    rouge_l_delta:    float = Field(0.0, description="ROUGE-L change (fine-tuned - baseline)")
    perplexity_delta: float = Field(0.0, description="Perplexity reduction (baseline - fine-tuned)")


class SamplePrediction(BaseModel):
    input:            str = Field(..., description="Input prompt")
    baseline_output:  str = Field(..., description="Baseline model response")
    finetuned_output: str = Field(..., description="Fine-tuned model response")
    reference:        str = Field(..., description="Ground-truth reference")


class AdapterInfo(BaseModel):
    size_mb:             float = Field(..., description="Adapter size in MB")
    trained_params_pct:  float = Field(..., description="% of params trained")
    training_time_mins:  float = Field(..., description="Training time in minutes")


# ── NEW: loss curve data point ────────────────────────────────────────────────
class LossPoint(BaseModel):
    """One epoch's train and eval loss — powers the loss curve chart."""
    epoch:      int   = Field(..., description="Epoch number")
    train_loss: float = Field(..., description="Training loss at this epoch")
    eval_loss:  float = Field(..., description="Validation loss at this epoch")


# ── NEW: dataset split sizes ──────────────────────────────────────────────────
class DatasetStats(BaseModel):
    """Row counts for each dataset split — shown in the dashboard stats panel."""
    train: int = Field(..., description="Training examples")
    val:   int = Field(..., description="Validation examples")
    test:  int = Field(..., description="Test examples")
    total: int = Field(..., description="Total examples in uploaded file")


class ResultsResponse(BaseModel):
    """Full evaluation results returned to the dashboard."""
    job_id:       str               = Field(..., description="Job UUID")
    model_id:     str               = Field(..., description="Base model used")
    baseline:     ModelMetrics      = Field(..., description="Baseline metrics")
    finetuned:    ModelMetrics      = Field(..., description="Fine-tuned metrics")
    improvement:  ImprovementMetrics = Field(..., description="Delta metrics")
    samples:      List[SamplePrediction] = Field(..., description="Sample comparisons")
    adapter:      AdapterInfo       = Field(..., description="Adapter metadata")
    loss_history: List[LossPoint]   = Field(default=[], description="Per-epoch loss curve data")
    dataset_stats: DatasetStats     = Field(
        default=DatasetStats(train=0, val=0, test=0, total=0),
        description="Dataset split sizes",
    )