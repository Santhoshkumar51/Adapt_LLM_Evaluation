import MetricsCard from "../components/MetricsCard";
import SampleComparison from "../components/SampleComparison";
import {
  adapterDownloadUrl,
} from "../api/adapteval";


/*
  Final evaluation dashboard for AdaptEval.
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


        <a
          className="download-adapter"
          href={adapterDownloadUrl(jobId)}
          download
        >
          <span>↓</span>
          Download LoRA adapter
        </a>

      </div>


      {/* Summary */}

      <div className="result-hero">

        <div className="result-hero-icon">
          ✓
        </div>


        <div>

          <div className="result-hero-title">
            Domain adaptation run completed
          </div>

          <div className="result-hero-text">
            The fine-tuned model is compared
            with the original base model on
            the same held-out evaluation set.
          </div>

        </div>


        <div className="result-hero-meta">

          <span>
            Held-out evaluation
          </span>

          <span>
            {adapter.trained_params_pct.toFixed(3)}
            % trainable
          </span>

        </div>

      </div>


      {/* Quantitative evaluation */}

      <div
        className="card results-card"
      >

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


      {/* Qualitative evaluation */}

      <div
        className="card results-card"
      >

        <div className="card-label">
          Qualitative evaluation
        </div>

        <div className="card-heading">
          Sample output comparison
        </div>

        <div className="card-description">

          Representative responses from
          the same held-out test set used
          for the quantitative evaluation.
          Five samples are shown when available.

        </div>


        <SampleComparison
          samples={samples}
        />

      </div>


      {/* Adapter information */}

      <div className="adapter-summary">

        <div>

          <div className="card-label">
            Reusable artifact
          </div>

          <div className="adapter-summary-title">
            LoRA adapter saved separately
          </div>

          <div className="adapter-summary-text">

            The trained adapter remains separate
            from the original Mistral-7B base model
            and can be retained as a compact
            reusable artifact.

          </div>

        </div>


        <a
          className="btn btn-outline"
          href={adapterDownloadUrl(jobId)}
          download
        >
          Download adapter
        </a>

      </div>


      {/* Reset */}

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