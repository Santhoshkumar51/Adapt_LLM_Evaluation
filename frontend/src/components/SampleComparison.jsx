import { useState } from "react";

/*
  Shows 5 side-by-side sample outputs: baseline vs fine-tuned.
  User can step through them with prev/next.
*/
export default function SampleComparison({ samples }) {
  const [idx, setIdx] = useState(0);
  const sample = samples[idx];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* ── Navigator ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        {samples.map((_, i) => (
          <button
            key={i}
            onClick={() => setIdx(i)}
            style={{
              width: 28, height: 28,
              borderRadius: "50%",
              border: i === idx ? "2px solid var(--accent)" : "1px solid var(--border)",
              background: i === idx ? "var(--accent-dim)" : "transparent",
              color: i === idx ? "var(--accent)" : "var(--text-muted)",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: 600,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}
          >
            {i + 1}
          </button>
        ))}
        <span style={{ fontSize: 12, color: "var(--text-muted)", marginLeft: "auto" }}>
          Example {idx + 1} of {samples.length}
        </span>
      </div>

      {/* ── Input ── */}
      <div style={{
        background: "var(--bg-surface)",
        borderRadius: "var(--radius-sm)",
        padding: "12px 16px",
        borderLeft: "3px solid var(--border)",
      }}>
        <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
          Input prompt
        </div>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, fontFamily: "var(--font-mono)" }}>
          {sample.input}
        </div>
      </div>

      {/* ── Side by side outputs ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {/* Baseline */}
        <div style={{
          background: "rgba(45,63,94,0.3)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          padding: "16px",
        }}>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 10 }}>
            Baseline model
          </div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7 }}>
            {sample.baseline_output || <em style={{ opacity: 0.4 }}>No output</em>}
          </div>
        </div>

        {/* Fine-tuned */}
        <div style={{
          background: "rgba(0,212,170,0.04)",
          border: "1px solid var(--border-accent)",
          borderRadius: "var(--radius-md)",
          padding: "16px",
        }}>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--accent)", marginBottom: 10 }}>
            Fine-tuned model
          </div>
          <div style={{ fontSize: 13, color: "var(--text-primary)", lineHeight: 1.7 }}>
            {sample.finetuned_output || <em style={{ opacity: 0.4 }}>No output</em>}
          </div>
        </div>
      </div>

      {/* Reference answer */}
      {sample.reference && (
        <div style={{
          background: "var(--bg-surface)",
          borderRadius: "var(--radius-sm)",
          padding: "12px 16px",
          borderLeft: "3px solid var(--warning)",
        }}>
          <div style={{ fontSize: 10, color: "var(--warning)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
            Reference answer
          </div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
            {sample.reference}
          </div>
        </div>
      )}
    </div>
  );
}