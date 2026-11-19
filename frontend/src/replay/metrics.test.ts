import { describe, expect, it } from "vitest";
import { EventSchema } from "../contracts/event";
import { deriveLiveMetrics } from "./metrics";
import { parseEventTrace } from "./trace";

const makeEvent = (
  seq: number,
  type: string,
  payload: Record<string, unknown> = {},
  t = seq / 10,
) =>
  EventSchema.parse({
    schema_version: 1,
    run_id: "trace-test",
    seq,
    t,
    type,
    node: null,
    level: null,
    version: null,
    payload,
  });

describe("event trace metrics", () => {
  it("parses valid JSONL and reports the line for invalid events", () => {
    const trace = [
      JSON.stringify(makeEvent(0, "run_start")),
      JSON.stringify(makeEvent(1, "search", { latency_ms: 2 })),
    ].join("\n");

    expect(parseEventTrace(trace)).toHaveLength(2);
    expect(() => parseEventTrace(`${trace}\nnot-json`)).toThrow("line 3");
  });

  it("derives query, repair, edge, connectivity, and elapsed metrics from events", () => {
    const events = [
      makeEvent(0, "run_start"),
      makeEvent(1, "search", { latency_ms: 2 }),
      makeEvent(2, "search", { latency_ms: 4 }),
      makeEvent(3, "pop"),
      makeEvent(4, "edge_add"),
      makeEvent(5, "edge_remove"),
      makeEvent(6, "metric", { name: "recall@10", value: 0.9 }),
      makeEvent(7, "metric", { name: "layer0_reachability", value: 0.98 }),
      makeEvent(8, "run_end", { elapsed_seconds: 2 }),
    ];

    expect(deriveLiveMetrics(events)).toMatchObject({
      recall: 0.9,
      meanLatencyMs: 3,
      queryCount: 2,
      qps: 1,
      repairCount: 1,
      modifiedEdges: 2,
      connectivity: 0.98,
      processingSeconds: 2,
    });
  });

  it("marks schema-example traces as synthetic", () => {
    const metrics = deriveLiveMetrics([
      makeEvent(0, "metric", { example: true, name: "recall@10", value: 0.96 }),
    ]);

    expect(metrics.synthetic).toBe(true);
  });
});
