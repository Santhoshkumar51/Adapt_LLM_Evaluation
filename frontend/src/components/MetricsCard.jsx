import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";

/*
  KPI comparison card.
  Shows accuracy, F1, perplexity for baseline vs fine-tuned,
  plus improvement deltas and efficiency stats.
*/
export default function MetricsCard({ baseline, finetuned, improvement, adapter }) {

  const chartData = [
    {
      metric: "Accuracy",
      Baseline:    parseFloat((baseline.accuracy  * 100).toFixed(1)),
      "Fine-tuned": parseFloat((finetuned.accuracy * 100).toFixed(1)),
    },
    {
      metric: "F1 Score",
      Baseline:    parseFloat((baseline.f1  * 100).toFixed(1)),
      "Fine-tuned": parseFloat((finetuned.f1 * 100).toFixed(1)),
    },
    {
      metric: "Perplexity ↓",
      Baseline:    parseFloat(baseline.perplexity.toFixed(2)),
      "Fine-tuned": parseFloat(finetuned.perplexity.toFixed(2)),
    },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* ── KPI numbers ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
        {[
          {
            label: "Accuracy",
            base: `${(baseline.accuracy  * 100).toFixed(1)}%`,
            tuned: `${(finetuned.accuracy * 100).toFixed(1)}%`,
            delta: `+${(improvement.accuracy_delta * 100).toFixed(1)}%`,
            positive: improvement.accuracy_delta > 0,
          },
          {
            label: "F1 Score",
            base: `${(baseline.f1  * 100).toFixed(1)}%`,
            tuned: `${(finetuned.f1 * 100).toFixed(1)}%`,
            delta: `+${(improvement.f1_delta * 100).toFixed(1)}%`,
            positive: improvement.f1_delta > 0,
          },
          {
            label: "Perplexity ↓",
            base:  baseline.perplexity.toFixed(2),
            tuned: finetuned.perplexity.toFixed(2),
            delta: `-${improvement.perplexity_delta.toFixed(2)}`,
            positive: improvement.perplexity_delta > 0,
          },
        ].map((kpi) => (
          <div key={kpi.label} style={{
            background: "var(--bg-surface)",
            borderRadius: "var(--radius-md)",
            padding: "16px 18px",
            border: "1px solid var(--border)",
          }}>
            <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 10 }}>
              {kpi.label}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <div>
                <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 2 }}>Baseline</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 16, color: "var(--text-secondary)" }}>
                  {kpi.base}
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 2 }}>Fine-tuned</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 16, color: "var(--accent)" }}>
                  {kpi.tuned}
                </div>
              </div>
            </div>
            <div style={{
              marginTop: 8,
              padding: "4px 8px",
              borderRadius: 4,
              background: kpi.positive ? "rgba(0,212,170,0.1)" : "rgba(239,68,68,0.1)",
              fontSize: 12,
              fontWeight: 600,
              color: kpi.positive ? "var(--success)" : "var(--error)",
              fontFamily: "var(--font-mono)",
              display: "inline-block",
            }}>
              {kpi.delta}
            </div>
          </div>
        ))}
      </div>

      {/* ── Bar chart ── */}
      <div style={{ height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} barGap={4}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
            <XAxis dataKey="metric" tick={{ fill: "#8b9ab8", fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#8b9ab8", fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: "#1a2236", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 8 }}
              labelStyle={{ color: "#f0f4ff", marginBottom: 4 }}
              itemStyle={{ color: "#8b9ab8" }}
            />
            <Legend
              wrapperStyle={{ fontSize: 12, color: "#8b9ab8", paddingTop: 12 }}
            />
            <Bar dataKey="Baseline"    fill="#2d3f5e" radius={[3,3,0,0]} />
            <Bar dataKey="Fine-tuned"  fill="#00d4aa" radius={[3,3,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Efficiency stats ── */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(3, 1fr)",
        gap: 12,
      }}>
        {[
          { label: "Adapter size",         value: `${adapter.size_mb.toFixed(1)} MB` },
          { label: "Params trained",        value: `${adapter.trained_params_pct.toFixed(3)}%` },
          { label: "Training time",         value: `${adapter.training_time_mins.toFixed(1)} min` },
        ].map((stat) => (
          <div key={stat.label} style={{
            background: "var(--bg-surface)",
            borderRadius: "var(--radius-sm)",
            padding: "12px 16px",
            border: "1px solid var(--border)",
          }}>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>{stat.label}</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 15, color: "var(--text-primary)", fontWeight: 500 }}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}