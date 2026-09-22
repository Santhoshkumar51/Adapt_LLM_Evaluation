import { useEffect, useState } from "react";
import { fetchModels } from "../api/adapteval";

/*
  Dropdown of supported base models fetched from GET /api/models.
  Calls onChange(modelId) when user selects a model.
*/
export default function ModelSelector({ value, onChange }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState(null);

  useEffect(() => {
    fetchModels()
      .then((data) => setModels(data.models))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Loading models...</div>
  );

  if (error) return (
    <div style={{ color: "var(--error)", fontSize: 13 }}>
      Could not load models: {error}
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">Select a base model</option>
        {models.map((m) => (
          <option key={m.model_id} value={m.model_id}>
            {m.display_name} ({m.parameters})
          </option>
        ))}
      </select>

      {/* Description of selected model */}
      {value && (() => {
        const selected = models.find((m) => m.model_id === value);
        return selected ? (
          <div style={{
            fontSize: 12,
            color: "var(--text-secondary)",
            padding: "8px 12px",
            background: "var(--bg-surface)",
            borderRadius: "var(--radius-sm)",
            borderLeft: "2px solid var(--border-accent)",
          }}>
            {selected.description}
          </div>
        ) : null;
      })()}
    </div>
  );
}