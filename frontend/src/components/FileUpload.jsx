import { useRef, useState } from "react";

const ALLOWED = [".jsonl", ".csv"];
const MAX_MB  = 50;

/*
  Drag-and-drop + click-to-browse file upload.
  Validates extension and size client-side before passing file up.
  Calls onChange(file | null).
*/
export default function FileUpload({ value, onChange }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError]       = useState(null);

  function validate(file) {
    setError(null);
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!ALLOWED.includes(ext)) {
      setError(`Unsupported file type. Upload a ${ALLOWED.join(" or ")} file.`);
      return false;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      setError(`File exceeds ${MAX_MB} MB limit.`);
      return false;
    }
    return true;
  }

  function handleFile(file) {
    if (validate(file)) onChange(file);
    else onChange(null);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        style={{
          border: `1.5px dashed ${dragging ? "var(--accent)" : value ? "var(--border-accent)" : "var(--border)"}`,
          borderRadius: "var(--radius-md)",
          padding: "28px",
          textAlign: "center",
          cursor: "pointer",
          background: dragging ? "var(--accent-dim)" : value ? "rgba(0,212,170,0.04)" : "var(--bg-surface)",
          transition: "all 0.15s ease",
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".jsonl,.csv"
          style={{ display: "none" }}
          onChange={(e) => { if (e.target.files[0]) handleFile(e.target.files[0]); }}
        />

        {value ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontSize: 20 }}>✓</span>
            <span style={{ color: "var(--accent)", fontWeight: 500, fontSize: 13 }}>
              {value.name}
            </span>
            <span style={{ color: "var(--text-muted)", fontSize: 12 }}>
              {(value.size / 1024).toFixed(1)} KB — click to change
            </span>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span style={{ fontSize: 24, opacity: 0.5 }}>↑</span>
            <span style={{ color: "var(--text-secondary)", fontSize: 13 }}>
              Drop your dataset here, or <span style={{ color: "var(--accent)" }}>browse</span>
            </span>
            <span style={{ color: "var(--text-muted)", fontSize: 12 }}>
              .jsonl or .csv · max 20 MB · up to 5,000 examples
            </span>
          </div>
        )}
      </div>

      {error && (
        <div style={{ color: "var(--error)", fontSize: 12, paddingLeft: 4 }}>
          {error}
        </div>
      )}
    </div>
  );
}