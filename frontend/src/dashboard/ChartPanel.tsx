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
              <stop offset="0%" stopColor="#a77aff" stopOpacity={0.28} />
              <stop offset="95%" stopColor="#a77aff" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#292531" strokeDasharray="3 5" vertical={false} />
          <XAxis dataKey="time" tick={{ fill: "#797486", fontSize: 10 }} tickLine={false} axisLine={false} />
          <YAxis tick={{ fill: "#797486", fontSize: 10 }} tickLine={false} axisLine={false} />
          <Tooltip contentStyle={{ background: "#17151d", border: "1px solid #393344", borderRadius: 8, color: "#eeeaf5" }} />
          <Area type="monotone" dataKey="displacement" stroke="#ad83ff" strokeWidth={2} fill="url(#movement-fill)" />
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  if (props.type === "priority") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={props.data} layout="vertical" margin={{ top: 4, right: 10, left: 2, bottom: 0 }}>
          <CartesianGrid stroke="#292531" strokeDasharray="3 5" horizontal={false} />
          <XAxis type="number" domain={[0, 100]} tick={{ fill: "#797486", fontSize: 10 }} tickLine={false} axisLine={false} />
          <YAxis type="category" dataKey="id" width={42} tick={{ fill: "#a9a3b4", fontSize: 10 }} tickLine={false} axisLine={false} />
          <Tooltip contentStyle={{ background: "#17151d", border: "1px solid #393344", borderRadius: 8, color: "#eeeaf5" }} />
          <Bar dataKey="priority" fill="#a77aff" radius={[0, 4, 4, 0]} barSize={13} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={props.data} margin={{ top: 8, right: 12, left: 0, bottom: 24 }}>
        <CartesianGrid stroke="#292531" strokeDasharray="3 5" vertical={false} />
        <XAxis dataKey="label" tick={{ fill: "#a9a3b4", fontSize: 9 }} tickLine={false} axisLine={false} interval={0} angle={-14} textAnchor="end" />
        <YAxis unit={props.unit} tick={{ fill: "#797486", fontSize: 10 }} tickLine={false} axisLine={false} />
        <Tooltip contentStyle={{ background: "#17151d", border: "1px solid #393344", borderRadius: 8, color: "#eeeaf5" }} />
        <Bar dataKey="value" fill="#a77aff" radius={[4, 4, 0, 0]} barSize={28} />
      </BarChart>
    </ResponsiveContainer>
  );
}
