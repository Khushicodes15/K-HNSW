import type { Event } from "../contracts/event";

export interface LiveMetricPoint {
  time: number;
  recall: number | null;
  latencyMs: number | null;
}

export interface LiveMetrics {
  runId: string | null;
  eventCount: number;
  recallName: string | null;
  recall: number | null;
  meanLatencyMs: number | null;
  queryCount: number;
  qps: number | null;
  repairCount: number;
  modifiedEdges: number;
  connectivity: number | null;
  processingSeconds: number | null;
  synthetic: boolean;
  timeline: LiveMetricPoint[];
}

const numericValue = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const metricName = (event: Event): string =>
  typeof event.payload.name === "string" ? event.payload.name.toLowerCase() : "";

export function deriveLiveMetrics(events: Event[]): LiveMetrics {
  const searchEvents = events.filter((event) => event.type === "search");
  const latencies = searchEvents
    .map((event) => numericValue(event.payload.latency_ms))
    .filter((value): value is number => value !== null && value >= 0);
  const metricEvents = events.filter((event) => event.type === "metric");
  const recallEvents = metricEvents.filter((event) => metricName(event).startsWith("recall@"));
  const connectivityEvents = metricEvents.filter((event) =>
    /connect|reachability/.test(metricName(event)),
  );
  const latestRecall = recallEvents.at(-1);
  const latestConnectivity = connectivityEvents.at(-1);
  const recallValue = latestRecall ? numericValue(latestRecall.payload.value) : null;
  const connectivityValue = latestConnectivity
    ? numericValue(latestConnectivity.payload.value)
    : null;
  const runEnd = [...events].reverse().find((event) => event.type === "run_end");
  const processingSeconds = runEnd
    ? numericValue(runEnd.payload.elapsed_seconds)
    : null;
  const queryCount = searchEvents.length;

  let latestTimelineRecall: number | null = null;
  const timeline = events
    .filter((event) => event.type === "search" || event.type === "metric")
    .map((event) => {
      if (event.type === "metric" && metricName(event).startsWith("recall@")) {
        latestTimelineRecall = numericValue(event.payload.value);
      }
      return {
        time: event.t,
        recall: latestTimelineRecall,
        latencyMs:
          event.type === "search" ? numericValue(event.payload.latency_ms) : null,
      };
    });

  return {
    runId: events.at(-1)?.run_id ?? null,
    eventCount: events.length,
    recall:
      recallValue !== null && recallValue >= 0 && recallValue <= 1
        ? recallValue
        : null,
    recallName: latestRecall ? metricName(latestRecall) : null,
    meanLatencyMs:
      latencies.length > 0
        ? latencies.reduce((sum, latency) => sum + latency, 0) / latencies.length
        : null,
    queryCount,
    qps:
      processingSeconds !== null && processingSeconds > 0
        ? queryCount / processingSeconds
        : null,
    repairCount: events.filter((event) => event.type === "pop").length,
    modifiedEdges: events.filter(
      (event) => event.type === "edge_add" || event.type === "edge_remove",
    ).length,
    connectivity:
      connectivityValue !== null &&
      connectivityValue >= 0 &&
      connectivityValue <= 1
        ? connectivityValue
        : null,
    processingSeconds:
      processingSeconds !== null && processingSeconds >= 0
        ? processingSeconds
        : null,
    synthetic: events.some((event) => event.payload.example === true),
    timeline,
  };
}
