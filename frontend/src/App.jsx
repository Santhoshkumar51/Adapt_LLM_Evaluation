import { useState } from "react";
import HomePage from "./pages/HomePage";
import TrainingPage from "./pages/TrainingPage";
import ResultsPage from "./pages/ResultsPage";

/*
  App manages global navigation state.
  Three screens: home → training → results.
  No router library needed — three-screen linear flow.
*/
export default function App() {
  const [screen, setScreen] = useState("home");  // "home" | "training" | "results"
  const [jobId, setJobId]   = useState(null);
  const [results, setResults] = useState(null);
  const [modelId, setModelId] = useState(null);

  function goToTraining(id, model) {
    setJobId(id);
    setModelId(model);
    setScreen("training");
  }

  function goToResults(data) {
    setResults(data);
    setScreen("results");
  }

  function reset() {
    setScreen("home");
    setJobId(null);
    setResults(null);
    setModelId(null);
  }

  return (
    <div className="app-shell">
      {/* ── Top bar ── */}
      <header className="topbar">
        <span className="topbar-logo">AdaptEval</span>
        <span className="topbar-sub">LoRA Fine-Tuning Platform</span>
        {screen !== "home" && (
          <button
            className="btn btn-outline"
            style={{ marginLeft: "auto", padding: "5px 14px", fontSize: 12 }}
            onClick={reset}
          >
            ← New run
          </button>
        )}
      </header>

      {/* ── Screens ── */}
      {screen === "home"     && <HomePage     onSubmit={goToTraining} />}
      {screen === "training" && <TrainingPage jobId={jobId} modelId={modelId} onComplete={goToResults} />}
      {screen === "results"  && <ResultsPage  results={results} modelId={modelId} jobId={jobId} onReset={reset} />}
    </div>
  );
}