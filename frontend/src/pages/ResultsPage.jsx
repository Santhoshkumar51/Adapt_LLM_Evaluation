import MetricsCard from "../components/MetricsCard";
import SampleComparison from "../components/SampleComparison";
import LossChart from "../LossChart";
import { adapterDownloadUrl } from "../api/adapteval";

/*
  Results dashboard.

  The page provides:
    - Quantitative baseline vs fine-tuned comparison
    - Dataset split statistics
    - Training loss curve
    - Qualitative sample comparison
    - One adapter download action
    - JSON results export
*/

function exportResults(results, jobId, modelId) {
  const payload = {
    job_id: jobId,
    model_id: modelId,
    exported_at: new Date().toISOString(),
    dataset_stats: results.dataset_stats,
    baseline: results.baseline,
    finetuned: results.finetuned,
    improvement: results.improvement,
    adapter: results.adapter,
    loss_history: results.loss_history,
  };

  const blob = new Blob(
    [JSON.stringify(payload, null, 2)],
    { type: "application/json" }
  );

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");

  a.href = url;
  a.download = `adapteval_results_${jobId?.slice(0, 8)}.json`;

  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  URL.revokeObjectURL(url);
}

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
    samples = [],
    adapter,
    loss_history = [],
    dataset_stats,
  } = results;

  return (
    <main className="page results-page">

      {/* Header */}
      <div className="results-header">
        <div>
          <div className="eyebrow">
            Evaluation complete
          </div>

          <h1 className="page-title">
            Results
          </h1>

          <p className="page-sub results-sub">
            {modelId?.split("/").pop()}
            {" · "}
            <span
              className="mono"
              style={{ fontSize: 12 }}
            >
              {jobId?.slice(0, 8)}
            </span>
          </p>
        </div>

        <div
          style={{
            display: "flex",
            gap: 10,
            alignItems: "flex-start",
            flexShrink: 0,
          }}
        >
          <button
            className="btn btn-outline"
            onClick={() =>
              exportResults(
                results,
                jobId,
                modelId
              )
            }
            style={{
              fontSize: 12,
              padding: "9px 14px",
            }}
          >
            ↗ Export results
          </button>

          <a
            className="download-adapter"
            href={adapterDownloadUrl(jobId)}
            download
          >
            <span>↓</span>
            Download adapter
          </a>
        </div>
      </div>

      {/* Completion banner */}
      <div className="result-hero">
        <div className="result-hero-icon">
          ✓
        </div>

        <div>
          <div className="result-hero-title">
            Domain adaptation run completed
          </div>

          <div className="result-hero-text">
            The fine-tuned model is compared with
            the original base model on the same
            held-out evaluation set.
          </div>
        </div>

        <div className="result-hero-meta">
          <span>
            Held-out evaluation
          </span>

          <span>
            {adapter.trained_params_pct.toFixed(3)}%
            {" "}trainable
          </span>
        </div>
      </div>

      {/* Dataset statistics */}
      {dataset_stats && (
        <div className="card results-card">
          <div className="card-label">
            Dataset
          </div>

          <div
            className="card-heading"
            style={{ marginBottom: 18 }}
          >
            Split statistics
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(4, 1fr)",
              gap: 12,
            }}
          >
            {[
              {
                label: "Total uploaded",
                value: dataset_stats.total,
              },
              {
                label: "Training split",
                value: dataset_stats.train,
              },
              {
                label: "Validation split",
                value: dataset_stats.val,
              },
              {
                label: "Test split",
                value: dataset_stats.test,
              },
            ].map((stat) => (
              <div
                key={stat.label}
                style={{
                  background:
                    "var(--bg-surface)",
                  border:
                    "1px solid var(--border)",
                  borderRadius:
                    "var(--radius-sm)",
                  padding: "12px 16px",
                }}
              >
                <div
                  style={{
                    fontSize: 10,
                    color:
                      "var(--text-muted)",
                    textTransform:
                      "uppercase",
                    letterSpacing:
                      "0.06em",
                    marginBottom: 6,
                  }}
                >
                  {stat.label}
                </div>

                <div
                  style={{
                    fontFamily:
                      "var(--font-mono)",
                    fontSize: 20,
                    fontWeight: 600,
                    color:
                      "var(--text-primary)",
                  }}
                >
                  {stat.value.toLocaleString()}
                </div>
              </div>
            ))}
          </div>

          <div
            style={{
              marginTop: 12,
              fontSize: 11,
              color: "var(--text-muted)",
            }}
          >
            80 / 10 / 10 train-validation-test
            split · baseline and fine-tuned models
            are evaluated on the same test split
          </div>
        </div>
      )}

      {/* Quantitative evaluation */}
      <div className="card results-card">
        <div className="card-heading-row">
          <div>
            <div className="card-label">
              Quantitative evaluation
            </div>

            <div className="card-heading">
              Baseline vs fine-tuned model
            </div>
          </div>

          <div className="metric-legend">
            <span>
              <i className="legend-dot baseline-dot" />
              Baseline
            </span>

            <span>
              <i className="legend-dot tuned-dot" />
              Fine-tuned
            </span>
          </div>
        </div>

        <MetricsCard
          baseline={baseline}
          finetuned={finetuned}
          improvement={improvement}
          adapter={adapter}
        />
      </div>

      {/* Training dynamics */}
      <div className="card results-card">
        <div className="card-label">
          Training dynamics
        </div>

        <div
          className="card-heading"
          style={{ marginBottom: 18 }}
        >
          Loss curve
        </div>

        <LossChart
          lossHistory={loss_history}
        />

        <div
          style={{
            marginTop: 12,
            fontSize: 11,
            color: "var(--text-muted)",
          }}
        >
          Training and validation loss recorded
          during fine-tuning.
        </div>
      </div>

      {/* Qualitative evaluation */}
      <div className="card results-card">
        <div className="card-label">
          Qualitative evaluation
        </div>

        <div className="card-heading">
          Sample output comparison
        </div>

        <div
          className="card-description"
          style={{
            margin: "6px 0 18px",
          }}
        >
          Representative responses from the same
          held-out test set.
        </div>

        <SampleComparison
          samples={samples}
        />
      </div>

      {/* Footer */}
      <div className="results-footer-action">
        <button
          className="btn btn-outline"
          onClick={onReset}
        >
          Run another fine-tune
        </button>
      </div>

    </main>
  );
}