/*
  Centralized API client for AdaptEval backend.
  All fetch calls live here — pages/components never call fetch directly.
  Base URL is empty string since Vite proxies /api → FastAPI at localhost:8000.
*/

const BASE = "/api";

async function handleResponse(res) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

/**
 * GET /api/models
 * Returns list of supported base models for the dropdown.
 */
export async function fetchModels() {
  const res = await fetch(`${BASE}/models`);
  return handleResponse(res);
}

/**
 * POST /api/finetune
 * Submits model selection + training data file.
 * Returns { job_id, status, message }.
 *
 * @param {string} modelId  - Hugging Face model ID
 * @param {File}   file     - .jsonl or .csv training data file
 */
export async function submitFinetuneJob(modelId, file) {
  const form = new FormData();
  form.append("model_id", modelId);
  form.append("file", file);

  const res = await fetch(`${BASE}/finetune`, {
    method: "POST",
    body: form,
    // Do NOT set Content-Type — browser sets multipart boundary automatically
  });
  return handleResponse(res);
}

/**
 * GET /api/results/{jobId}/status
 * Polls training progress. Returns { job_id, status, progress, error }.
 *
 * @param {string} jobId
 */
export async function fetchJobStatus(jobId) {
  const res = await fetch(`${BASE}/results/${jobId}/status`);
  return handleResponse(res);
}

/**
 * GET /api/results/{jobId}
 * Returns full evaluation results once job is COMPLETE.
 *
 * @param {string} jobId
 */
export async function fetchResults(jobId) {
  const res = await fetch(`${BASE}/results/${jobId}`);
  return handleResponse(res);
}

/**
 * Returns the download URL for the adapter file.
 * Called directly as an anchor href — not a fetch call.
 *
 * @param {string} jobId
 */
export function adapterDownloadUrl(jobId) {
  return `${BASE}/results/${jobId}/download`;
}