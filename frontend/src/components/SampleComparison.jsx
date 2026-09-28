import { useState } from "react";

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  }

  return (
    <button
      onClick={handleCopy}
      title="Copy to clipboard"
      style={{
        background: "transparent",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-sm)",
        color: copied ? "var(--accent)" : "var(--text-muted)",
        cursor: "pointer",
        fontSize: 11,
        padding: "2px 8px",
        fontFamily: "var(--font-mono)",
        transition: "all 0.15s",
        flexShrink: 0,
      }}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export default function SampleComparison({ samples = [] }) {
  const [idx, setIdx] = useState(0);

  if (samples.length === 0) {
    return (
      <div style={{
        padding: "24px",
        textAlign: "center",
        background: "var(--bg-surface)",
        borderRadius: "var(--radius-sm)",
        color: "var(--text-muted)",
        fontSize: 13,
      }}>
        Sample predictions will appear here after a real fine-tuning and evaluation run.
      </div>
    );
  }

  const sample = samples[idx];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Navigator */}
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

      {/* Input prompt */}
      <div style={{
        background: "var(--bg-surface)",
        borderRadius: "var(--radius-sm)",
        padding: "12px 16px",
        borderLeft: "3px solid var(--border)",
      }}>
        <div style={{
          fontSize: 10, color: "var(--text-muted)",
          textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6,
        }}>
          Input prompt
        </div>
        <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, fontFamily: "var(--font-mono)" }}>
          {sample.input}
        </div>
      </div>

      {/* Side by side outputs */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>

        {/* Baseline */}
        <div style={{
          background: "rgba(45,63,94,0.3)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          padding: "16px",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <div style={{
              fontSize: 10, fontWeight: 600, letterSpacing: "0.08em",
              textTransform: "uppercase", color: "var(--text-muted)",
            }}>
              Baseline model
            </div>
            <CopyButton text={sample.baseline_output || ""} />
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
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <div style={{
              fontSize: 10, fontWeight: 600, letterSpacing: "0.08em",
              textTransform: "uppercase", color: "var(--accent)",
            }}>
              Fine-tuned model
            </div>
            <CopyButton text={sample.finetuned_output || ""} />
          </div>
          <div style={{ fontSize: 13, color: "var(--text-primary)", lineHeight: 1.7 }}>
            {sample.finetuned_output || <em style={{ opacity: 0.4 }}>No output</em>}
          </div>
        </div>

      </div>

      {/* Reference */}
      {sample.reference && (
        <div style={{
          background: "var(--bg-surface)",
          borderRadius: "var(--radius-sm)",
          padding: "12px 16px",
          borderLeft: "3px solid var(--warning)",
        }}>
          <div style={{
            fontSize: 10, color: "var(--warning)",
            textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6,
          }}>
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