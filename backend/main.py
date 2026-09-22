"""
AdaptEval FastAPI application entry point.
Mounts all route groups, configures CORS for the React frontend,
and manages the job store shared across routes.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import models, finetune, results

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── In-memory job store ───────────────────────────────────────────────────────
# Stores status and results for each fine-tuning job by job_id (UUID).
# In production this would be a Redis store or a database.
JOB_STORE: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    logger.info("AdaptEval backend starting up...")
    yield
    logger.info("AdaptEval backend shutting down...")
    JOB_STORE.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AdaptEval API",
    description="Automated LoRA fine-tuning and evaluation backend for AdaptEval.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow requests from the React dev server during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # Vite default dev port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Route registration ────────────────────────────────────────────────────────
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(finetune.router, prefix="/api/finetune", tags=["Fine-Tuning"])
app.include_router(results.router, prefix="/api/results", tags=["Results"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Simple liveness probe — returns OK if the server is running."""
    return {"status": "ok", "service": "AdaptEval API"}