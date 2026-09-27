/*
  Performance comparison for the final AdaptEval evaluation setup.

  Metrics:
    - ROUGE-L
    - Perplexity

  Accuracy/F1 are intentionally not shown.
*/

export default function MetricsCard({
  baseline,
  finetuned,
  improvement,
  adapter,
}) {

  const metrics = [
    {
      key: "rouge",
      label: "ROUGE-L",
      description:
        "Reference-based lexical overlap",

      base: baseline.rouge_l,
      tuned: finetuned.rouge_l,

      delta:
        improvement.rouge_l_delta,

      format: (value) =>
        value.toFixed(4),

      deltaText:
        `+${improvement.rouge_l_delta.toFixed(4)}`,

      max:
        Math.max(
          baseline.rouge_l,
          finetuned.rouge_l,
          0.0001
        ),
    },

    {
      key: "perplexity",
      label: "Perplexity",
      description:
        "Lower indicates better sequence likelihood",

      base: baseline.perplexity,
      tuned: finetuned.perplexity,

      delta:
        improvement.perplexity_delta,

      format: (value) =>
        value.toFixed(4),

      deltaText:
        `-${improvement.perplexity_delta.toFixed(4)}`,

      max:
        Math.max(
          baseline.perplexity,
          finetuned.perplexity,
          0.0001
        ),
    },
  ];

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 24,
      }}
    >

      {/* Metric panels */}

      <div className="metric-grid">

        {metrics.map((metric) => (

          <div
            className="metric-panel"
            key={metric.key}
          >

            <div className="metric-panel-head">

              <div>

                <div className="metric-title">
                  {metric.label}
                </div>

                <div className="metric-description">
                  {metric.description}
                </div>

              </div>

              <span className="metric-delta">
                {metric.deltaText}
              </span>

            </div>


            <div className="metric-values">

              <div>

                <div className="metric-label">
                  Baseline
                </div>

                <div className="metric-value metric-value-muted">
                  {metric.format(metric.base)}
                </div>

              </div>


              <div
                style={{
                  textAlign: "right",
                }}
              >

                <div className="metric-label">
                  Fine-tuned
                </div>

                <div className="metric-value">
                  {metric.format(metric.tuned)}
                </div>

              </div>

            </div>


            <div className="metric-bars">

              <div className="metric-bar-row">

                <span className="metric-bar-label">
                  Base
                </span>

                <div className="metric-bar-track">

                  <div
                    className="metric-bar baseline-bar"
                    style={{
                      width: `${Math.max(
                        (metric.base / metric.max) * 100,
                        2
                      )}%`,
                    }}
                  />

                </div>

              </div>


              <div className="metric-bar-row">

                <span className="metric-bar-label">
                  Tuned
                </span>

                <div className="metric-bar-track">

                  <div
                    className="metric-bar tuned-bar"
                    style={{
                      width: `${Math.max(
                        (metric.tuned / metric.max) * 100,
                        2
                      )}%`,
                    }}
                  />

                </div>

              </div>

            </div>

          </div>

        ))}

      </div>


      {/* Efficiency */}

      <div className="efficiency-grid">

        {[
          {
            label: "Adapter artifact",
            value:
              `${adapter.size_mb.toFixed(1)} MB`,
          },

          {
            label: "Trainable parameters",
            value:
              `${adapter.trained_params_pct.toFixed(3)}%`,
          },

          {
            label: "Training time",
            value:
              `${adapter.training_time_mins.toFixed(1)} min`,
          },

        ].map((stat) => (

          <div
            className="efficiency-stat"
            key={stat.label}
          >

            <div className="metric-label">
              {stat.label}
            </div>

            <div className="efficiency-value">
              {stat.value}
            </div>

          </div>

        ))}

      </div>


      {/* Evaluation note */}

      <div className="evaluation-note">

        <span className="evaluation-note-icon">
          i
        </span>

        <div>

          <strong>
            Evaluation note
          </strong>

          <span>
            ROUGE-L measures reference overlap
            and perplexity measures prediction
            likelihood on the evaluated sequences.
            These metrics do not directly establish
            medical factuality or clinical reliability.
          </span>

        </div>

      </div>

    </div>
  );
}