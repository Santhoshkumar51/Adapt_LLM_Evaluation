import { useEffect, useRef, useState } from "react";
import ProgressTracker from "../components/ProgressTracker";
import { fetchJobStatus, fetchResults } from "../api/adapteval";

const POLL_INTERVAL_MS = 12000;

/*
  Screen 2 — polls /status every 12 seconds.
  Transitions to ResultsPage automatically when status === "complete".
*/

export default function TrainingPage({ jobId, modelId, onComplete }) {
  const [status, setStatus] = useState("queued");
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);

  const intervalRef = useRef(null);

  useEffect(() => {
    let isMounted = true;

    async function poll() {
      try {
        const data = await fetchJobStatus(jobId);

        if (!isMounted) return;

        setStatus(data.status);
        setProgress(data.progress || null);
        setError(data.error || null);

        if (data.status === "complete") {
          clearInterval(intervalRef.current);

          const results = await fetchResults(jobId);

          if (isMounted) {
            onComplete(results);
          }
        }

        if (data.status === "failed") {
          clearInterval(intervalRef.current);
        }

      } catch (e) {
        if (!isMounted) return;

        setError(e.message || "Failed to fetch job status.");
        clearInterval(intervalRef.current);
      }
    }

    // Immediate first poll
    poll();

    // Continue polling
    intervalRef.current = setInterval(
      poll,
      POLL_INTERVAL_MS
    );

    return () => {
      isMounted = false;
      clearInterval(intervalRef.current);
    };
  }, [jobId, onComplete]);

  const statusMessages = {
    queued:
      "Your job is queued. Training will begin shortly.",

    preparing:
      "Preprocessing dataset and applying prompt template...",

    training:
      "Fine-tuning in progress. This takes 15–45 minutes.",

    evaluating:
      "Evaluating baseline and fine-tuned model on test set...",

    complete:
      "Complete. Loading your results...",

    failed:
      "Job failed. See error below.",
  };

  const stages = [
    {
      stage: "queued",
      desc:
        "Job accepted — model will be fetched from Hugging Face Hub if not cached.",
    },
    {
      stage: "preparing",
      desc:
        "Dataset split into 80/10/10 train/val/test. Prompt template applied. Tokenized.",
    },
    {
      stage: "training",
      desc:
        "LoRA adapters injected. Fine-tuning with QLoRA for configured epochs.",
    },
    {
      stage: "evaluating",
      desc:
        "Both models run on test split. Accuracy, F1, perplexity computed.",
    },
    {
      stage: "complete",
      desc:
        "Adapter weights merged and saved. Dashboard ready.",
    },
  ];

  return (
    <main className="page">

      <h1 className="page-title">
        Fine-tuning in progress
      </h1>

      <p className="page-sub">
        Job{" "}
        <span
          className="mono"
          style={{
            color: "var(--accent)",
            fontSize: 13,
          }}
        >
          {jobId?.slice(0, 8)}
        </span>

        {" · "}

        {modelId?.split("/").pop()}
      </p>

      <div
        className="card"
        style={{ marginBottom: 20 }}
      >
        <div className="card-label">
          Pipeline status
        </div>

        <ProgressTracker
          status={status}
          progress={progress}
          error={error}
        />
      </div>

      <div
        style={{
          fontSize: 13,
          color: "var(--text-secondary)",
          textAlign: "center",
          padding: "16px",
          lineHeight: 1.6,
        }}
      >
        {statusMessages[status] || "Processing..."}
      </div>

      {status !== "failed" && (
        <div
          style={{
            marginTop: 24,
            display: "flex",
            flexDirection: "column",
            gap: 8,
            background: "var(--bg-surface)",
            borderRadius: "var(--radius-md)",
            padding: "16px 20px",
          }}
        >

          <div className="section-label">
            What's happening now
          </div>

          {stages.map(({ stage, desc }) => (
            <div
              key={stage}
              style={{
                display: "flex",
                gap: 10,
                alignItems: "flex-start",
                opacity:
                  status === stage
                    ? 1
                    : stages.findIndex(
                        (s) => s.stage === status
                      ) >
                      stages.findIndex(
                        (s) => s.stage === stage
                      )
                    ? 0.5
                    : 0.3,
              }}
            >

              <span
                style={{
                  color: "var(--accent)",
                  fontSize: 12,
                  marginTop: 1,
                }}
              >
                ›
              </span>

              <span
                style={{
                  fontSize: 12,
                  color: "var(--text-secondary)",
                  lineHeight: 1.6,
                }}
              >
                <strong
                  style={{
                    color: "var(--text-primary)",
                    textTransform: "capitalize",
                  }}
                >
                  {stage}
                </strong>
                : {desc}
              </span>

            </div>
          ))}

        </div>
      )}

    </main>
  );
}