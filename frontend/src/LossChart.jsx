import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

/*
  Training loss curve — shows train_loss and eval_loss per epoch.
  Renders nothing if loss_history is empty (test mode with no real training).
*/
export default function LossChart({ lossHistory = [] }) {
  if (!lossHistory || lossHistory.length === 0) {
    return (
      <div style={{
        padding: "24px",
        textAlign: "center",
        background: "var(--bg-surface)",
        borderRadius: "var(--radius-sm)",
        color: "var(--text-muted)",
        fontSize: 13,
      }}>
        Loss curve will appear here after a real training run.
      </div>
    );
  }

  return (
    <div style={{ height: 220 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={lossHistory}
          margin={{ top: 4, right: 12, left: -10, bottom: 0 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.04)"
            vertical={false}
          />
          <XAxis
            dataKey="epoch"
            tick={{ fill: "#8b9ab8", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            label={{
              value: "Epoch",
              position: "insideBottom",
              offset: -2,
              fill: "#5b6982",
              fontSize: 11,
            }}
          />
          <YAxis
            tick={{ fill: "#8b9ab8", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <Tooltip
            contentStyle={{
              background: "#1a2236",
              border: "1px solid rgba(255,255,255,0.07)",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#f0f4ff", marginBottom: 4 }}
            itemStyle={{ color: "#8b9ab8" }}
            formatter={(value) => value.toFixed(4)}
            labelFormatter={(label) => `Epoch ${label}`}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "#8b9ab8", paddingTop: 12 }}
          />
          <Line
            type="monotone"
            dataKey="train_loss"
            name="Train loss"
            stroke="#50627e"
            strokeWidth={2}
            dot={{ r: 3, fill: "#50627e" }}
            activeDot={{ r: 5 }}
          />
          <Line
            type="monotone"
            dataKey="eval_loss"
            name="Eval loss"
            stroke="#00d4aa"
            strokeWidth={2}
            dot={{ r: 3, fill: "#00d4aa" }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}