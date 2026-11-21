import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ScheduledRepair } from "../data/playground";

interface MovementPoint {
  time: number;
  displacement: number;
  moved: number;
}

interface BaselinePoint {
  label: string;
  value: number;
}

type ChartPanelProps =
  | { type: "movement"; data: MovementPoint[] }
  | { type: "priority"; data: ScheduledRepair[] }
  | { type: "baseline"; data: BaselinePoint[]; unit: string };

export default function ChartPanel(props: ChartPanelProps) {
  if (props.type === "movement") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={props.data} margin={{ top: 10, right: 4, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="movement-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#7c3aed" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#7c3aed" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#e4e7ec" strokeDasharray="3 5" vertical={false} />
          <XAxis dataKey="time" tick={{ fill: "#667085", fontSize: 10 }} tickLine={false} axisLine={false} />
          <YAxis tick={{ fill: "#667085", fontSize: 10 }} tickLine={false} axisLine={false} />
          <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #e4e7ec", borderRadius: 8, color: "#263047" }} />
          <Area type="monotone" dataKey="displacement" stroke="#7c3aed" strokeWidth={2} fill="url(#movement-fill)" />
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  if (props.type === "priority") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={props.data} layout="vertical" margin={{ top: 4, right: 10, left: 2, bottom: 0 }}>
          <CartesianGrid stroke="#e4e7ec" strokeDasharray="3 5" horizontal={false} />
          <XAxis type="number" domain={[0, 100]} tick={{ fill: "#667085", fontSize: 10 }} tickLine={false} axisLine={false} />
          <YAxis type="category" dataKey="id" width={42} tick={{ fill: "#667085", fontSize: 10 }} tickLine={false} axisLine={false} />
          <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #e4e7ec", borderRadius: 8, color: "#263047" }} />
          <Bar dataKey="priority" fill="#7c3aed" radius={[0, 4, 4, 0]} barSize={13} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={props.data} margin={{ top: 8, right: 12, left: 0, bottom: 24 }}>
        <CartesianGrid stroke="#e4e7ec" strokeDasharray="3 5" vertical={false} />
        <XAxis dataKey="label" tick={{ fill: "#667085", fontSize: 9 }} tickLine={false} axisLine={false} interval={0} angle={-14} textAnchor="end" />
        <YAxis unit={props.unit} tick={{ fill: "#667085", fontSize: 10 }} tickLine={false} axisLine={false} />
        <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #e4e7ec", borderRadius: 8, color: "#263047" }} />
        <Bar dataKey="value" fill="#7c3aed" radius={[4, 4, 0, 0]} barSize={28} />
      </BarChart>
    </ResponsiveContainer>
  );
}
