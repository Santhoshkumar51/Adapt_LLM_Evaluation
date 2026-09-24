import MetricsCard from "../components/MetricsCard";
import SampleComparison from "../components/SampleComparison";
import { adapterDownloadUrl } from "../api/adapteval";

/*
  Screen 3 — full results dashboard.
  KPI comparison, sample outputs, adapter download.
*/

export default function ResultsPage({
  results,
  modelId,
  jobId,
  onReset,
}) {
  const {
    baseline,
    finetuned,
    improvement,
    samples,
    adapter,
  } = results;

  return (
    <main className="page">

      {/* ── Header ── */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          marginBottom: 36,
        }}
      >
        <div>
          <h1 className="page-title">
            Results
          </h1>

          <p
            className="page-sub"
            style={{ marginBottom: 0 }}
          >
            {modelId?.split("/").pop()} ·{" "}
            <span
              className="mono"
              style={{ fontSize: 12 }}
            >
              {jobId?.slice(0, 8)}
            </span>
          </p>
        </div>

        {/* Download adapter */}
        <a
          href={adapterDownloadUrl(jobId)}
          download
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: "9px 16px",
            borderRadius: "var(--radius-sm)",
            background: "var(--accent-dim)",
            border: "1px solid var(--border-accent)",
            color: "var(--accent)",
            textDecoration: "none",
            fontSize: 13,
            fontWeight: 500,
            transition: "background 0.15s",
          }}
        >
          ↓ Download adapter
        </a>
      </div>

      {/* ── Summary banner ── */}
      <div
        style={{
          background: "rgba(0,212,170,0.06)",
          border: "1px solid var(--border-accent)",
          borderRadius: "var(--radius-md)",
          padding: "16px 20px",
          marginBottom: 24,
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <span style={{ fontSize: 20 }}>
          ✓
        </span>

        <div>
          <div
            style={{
              fontWeight: 600,
              fontSize: 14,
              color: "var(--accent)",
              marginBottom: 2,
            }}
          >
            Fine-tuning improved accuracy by{" "}
            {(improvement.accuracy_delta * 100).toFixed(1)}%
          </div>

          <div
            style={{
              fontSize: 12,
              color: "var(--text-secondary)",
            }}
          >
            Adapter size:{" "}
            {adapter.size_mb.toFixed(1)} MB ·{" "}
            {adapter.trained_params_pct.toFixed(3)}%
            {" "}of parameters trained ·{" "}
            {adapter.training_time_mins.toFixed(1)}
            {" "}min training time
          </div>
        </div>
      </div>

      {/* ── KPI metrics ── */}
      <div
        className="card"
        style={{ marginBottom: 20 }}
      >
        <div className="card-label">
          Performance comparison
        </div>

        <MetricsCard
          baseline={baseline}
          finetuned={finetuned}
          improvement={improvement}
          adapter={adapter}
        />
      </div>

      {/* ── Sample outputs ── */}
      <div
        className="card"
        style={{ marginBottom: 20 }}
      >
        <div className="card-label">
          Sample output comparison
        </div>

        <div
          style={{
            fontSize: 12,
            color: "var(--text-muted)",
            marginBottom: 16,
          }}
        >
          Same input, different models — same test set
          your evaluation metrics are computed on.
        </div>

        <SampleComparison
          samples={samples}
        />
      </div>

      {/* ── Run another ── */}
      <div
        style={{
          textAlign: "center",
          paddingTop: 16,
        }}
      >
        <button
          className="btn btn-outline"
          onClick={onReset}
          style={{ padding: "10px 28px" }}
        >
          Run another fine-tune
        </button>
      </div>

    </main>
  );
}