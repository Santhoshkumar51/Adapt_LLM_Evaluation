import {
  useEffect,
  useRef,
  useState,
} from "react";

import ProgressTracker from "../components/ProgressTracker";

import {
  fetchJobStatus,
  fetchResults,
} from "../api/adapteval";


const POLL_INTERVAL_MS = 12000;


/*
  Screen 2 — polls the backend until
  evaluation is complete.
*/

export default function TrainingPage({
  jobId,
  modelId,
  onComplete,
}) {

  const [status, setStatus] =
    useState("queued");

  const [progress, setProgress] =
    useState(null);

  const [error, setError] =
    useState(null);

  const intervalRef =
    useRef(null);


  useEffect(() => {

    let isMounted = true;


    async function poll() {

      try {

        const data =
          await fetchJobStatus(jobId);

        if (!isMounted)
          return;


        setStatus(data.status);

        setProgress(
          data.progress || null
        );

        setError(
          data.error || null
        );


        if (
          data.status === "complete"
        ) {

          clearInterval(
            intervalRef.current
          );

          const results =
            await fetchResults(jobId);

          if (isMounted) {
            onComplete(results);
          }

        }


        if (
          data.status === "failed"
        ) {

          clearInterval(
            intervalRef.current
          );

        }

      } catch (e) {

        if (!isMounted)
          return;

        setError(
          e.message ||
          "Failed to fetch job status."
        );

        clearInterval(
          intervalRef.current
        );

      }

    }


    poll();


    intervalRef.current =
      setInterval(
        poll,
        POLL_INTERVAL_MS
      );


    return () => {

      isMounted = false;

      clearInterval(
        intervalRef.current
      );

    };

  }, [
    jobId,
    onComplete,
  ]);


  const statusMessages = {

    queued:
      "Your job is queued. Training will begin shortly.",

    preparing:
      "Preparing the dataset, prompt format, and evaluation splits...",

    training:
      "Fine-tuning is in progress. Duration depends on the selected model and hardware.",

    evaluating:
      "Evaluating the baseline and fine-tuned models on the held-out test set...",

    complete:
      "Complete. Loading your evaluation results...",

    failed:
      "Job failed. See the error details below.",
  };


  const stages = [

    {
      stage: "queued",
      desc:
        "Job accepted and the selected base model is prepared.",
    },

    {
      stage: "preparing",
      desc:
        "Dataset is formatted and split into 80/10/10 train, validation, and test sets.",
    },

    {
      stage: "training",
      desc:
        "LoRA adapters are injected and QLoRA fine-tuning is performed.",
    },

    {
      stage: "evaluating",
      desc:
        "ROUGE-L and perplexity are computed on the held-out test set.",
    },

    {
      stage: "complete",
      desc:
        "The LoRA adapter is saved separately and the dashboard is ready.",
    },

  ];


  const currentIndex =
    stages.findIndex(
      (item) =>
        item.stage === status
    );


  return (

    <main className="page training-page">

      <div className="eyebrow">
        AdaptEval pipeline
      </div>


      <h1 className="page-title">
        Fine-tuning in progress
      </h1>


      <p className="page-sub">

        Job{" "}

        <span
          className="mono training-job-id"
        >
          {jobId?.slice(0, 8)}
        </span>

        {" · "}

        {modelId?.split("/").pop()}

      </p>


      <div
        className="card training-status-card"
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


      <div className="training-message">

        <div
          className="training-message-dot"
        />

        <span>
          {statusMessages[status] ||
            "Processing..."}
        </span>

      </div>


      {status !== "failed" && (

        <div className="pipeline-card">

          <div className="section-label">
            What's happening now
          </div>


          {stages.map(
            ({
              stage,
              desc,
            }, index) => {

              const active =
                status === stage;

              const done =
                currentIndex > index;


              return (

                <div
                  key={stage}
                  className={
                    `pipeline-stage ` +
                    `${active ? "active" : ""} ` +
                    `${done ? "done" : ""}`
                  }
                >

                  <div
                    className="pipeline-stage-marker"
                  >
                    {done
                      ? "✓"
                      : index + 1}
                  </div>


                  <div>

                    <div
                      className="pipeline-stage-title"
                    >
                      {stage}
                    </div>

                    <div
                      className="pipeline-stage-desc"
                    >
                      {desc}
                    </div>

                  </div>

                </div>

              );

            }
          )}

        </div>

      )}

    </main>
  );
}