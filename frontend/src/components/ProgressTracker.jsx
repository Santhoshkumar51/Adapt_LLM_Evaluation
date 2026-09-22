/*
  Displays training lifecycle: queued → preparing → training → evaluating → complete.
  Shows epoch progress and live loss values during training state.
*/

const STAGES = ["queued", "preparing", "training", "evaluating", "complete"];

function stageIndex(status) {
  return STAGES.indexOf(status);
}

export default function ProgressTracker({ status, progress, error }) {
  const currentIdx = stageIndex(status);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>

      {/* ── Stage indicators ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
        {STAGES.map((stage, i) => {
          const done    = i < currentIdx;
          const active  = i === currentIdx;
          const pending = i > currentIdx;
          return (
            <div key={stage} style={{ display: "flex", alignItems: "center", flex: i < STAGES.length - 1 ? 1 : "none" }}>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
                <div style={{
                  width: 28, height: 28,
                  borderRadius: "50%",
                  border: `2px solid ${done ? "var(--accent)" : active ? "var(--accent)" : "var(--border)"}`,
                  background: done ? "var(--accent)" : active ? "var(--accent-dim)" : "transparent",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, fontWeight: 600,
                  color: done ? "#0b0f1a" : active ? "var(--accent)" : "var(--text-muted)",
                  transition: "all 0.3s ease",
                  position: "relative",
                }}>
                  {done ? "✓" : i + 1}
                  {active && (
                    <span style={{
                      position: "absolute",
                      inset: -3,
                      borderRadius: "50%",
                      border: "2px solid var(--accent)",
                      opacity: 0.4,
                      animation: "pulse 1.5s infinite",
                    }} />
                  )}
                </div>
                <span style={{
                  fontSize: 10,
                  fontWeight: 500,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  color: active ? "var(--accent)" : done ? "var(--text-secondary)" : "var(--text-muted)",
                  whiteSpace: "nowrap",
                }}>
                  {stage}
                </span>
              </div>
              {i < STAGES.length - 1 && (
                <div style={{
                  flex: 1,
                  height: 2,
                  marginBottom: 18,
                  background: i < currentIdx ? "var(--accent)" : "var(--border)",
                  transition: "background 0.4s ease",
                }} />
              )}
            </div>
          );
        })}
      </div>

      {/* ── Training progress detail ── */}
      {status === "training" && progress && (
        <div style={{
          background: "var(--bg-surface)",
          borderRadius: "var(--radius-md)",
          padding: "20px 24px",
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>
              Epoch {progress.current_epoch} of {progress.total_epochs}
            </span>
            <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
              {progress.elapsed_mins.toFixed(1)} min elapsed
            </span>
          </div>

          {/* Epoch progress bar */}
          <div style={{ height: 4, background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
            <div style={{
              height: "100%",
              width: `${(progress.current_epoch / progress.total_epochs) * 100}%`,
              background: "var(--accent)",
              borderRadius: 2,
              transition: "width 0.5s ease",
            }} />
          </div>

          {/* Loss values */}
          <div style={{ display: "flex", gap: 24 }}>
            <div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 2 }}>Train loss</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 15, color: "var(--text-primary)" }}>
                {progress.train_loss.toFixed(4)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 2 }}>Eval loss</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 15, color: "var(--accent)" }}>
                {progress.eval_loss.toFixed(4)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Error state ── */}
      {status === "failed" && error && (
        <div style={{
          background: "rgba(239,68,68,0.08)",
          border: "1px solid rgba(239,68,68,0.2)",
          borderRadius: "var(--radius-md)",
          padding: "16px 20px",
          color: "var(--error)",
          fontSize: 13,
        }}>
          <strong>Job failed:</strong> {error}
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { transform: scale(1); opacity: 0.4; }
          50%       { transform: scale(1.4); opacity: 0; }
        }
      `}</style>
    </div>
  );
}