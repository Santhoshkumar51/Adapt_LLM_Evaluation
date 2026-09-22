import { useState } from "react";
import ModelSelector from "../components/ModelSelector";
import FileUpload    from "../components/FileUpload";
import { submitFinetuneJob } from "../api/adapteval";

/*
  Screen 1 — model selection + dataset upload.
  On submit, calls POST /api/finetune and navigates to TrainingPage.
*/
export default function HomePage({ onSubmit }) {
  const [modelId, setModelId] = useState("");
  const [file,    setFile]    = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const canSubmit = modelId && file && !loading;

  async function handleSubmit() {
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    try {
      const data = await submitFinetuneJob(modelId, file);
      onSubmit(data.job_id, modelId);
    } catch (e) {
      setError(e.message);
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <h1 className="page-title">Fine-tune your LLM</h1>
      <p className="page-sub">
        Upload domain data, pick a base model — AdaptEval handles everything else
        and shows you exactly how much better your model gets.
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

        {/* Model selection */}
        <div className="card">
          <div className="card-label">Base model</div>
          <ModelSelector value={modelId} onChange={setModelId} />
        </div>

        {/* Data upload */}
        <div className="card">
          <div className="card-label">Training dataset</div>
          <FileUpload value={file} onChange={setFile} />
          <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-muted)", lineHeight: 1.6 }}>
            Your file should be a <span className="mono">.jsonl</span> with one JSON object per line,
            each containing <span className="mono">"question"</span> and <span className="mono">"answer"</span> keys.
            CSV with the same columns also accepted.
          </div>
        </div>

        {/* Format example */}
        <div style={{
          background: "var(--bg-surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          padding: "16px 20px",
        }}>
          <div className="section-label">Expected format (.jsonl)</div>
          <pre style={{
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "var(--text-secondary)",
            lineHeight: 1.8,
            overflow: "auto",
          }}>
{`{"question": "What is LoRA?", "answer": "LoRA is a parameter-efficient fine-tuning method that trains low-rank adapter matrices instead of updating full model weights."}
{"question": "What is QLoRA?", "answer": "QLoRA extends LoRA with 4-bit quantization, enabling fine-tuning of large models on single consumer GPUs."}`}
          </pre>
        </div>

        {/* Error */}
        {error && (
          <div style={{
            background: "rgba(239,68,68,0.08)",
            border: "1px solid rgba(239,68,68,0.2)",
            borderRadius: "var(--radius-sm)",
            padding: "12px 16px",
            color: "var(--error)",
            fontSize: 13,
          }}>
            {error}
          </div>
        )}

        {/* Submit */}
        <button className="btn btn-primary" disabled={!canSubmit} onClick={handleSubmit}>
          {loading ? "Submitting..." : "Start fine-tuning →"}
        </button>
      </div>
    </main>
  );
}