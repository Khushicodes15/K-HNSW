import { z } from "zod";

/**
 * Enumeration of kinetic HNSW trace event types, matching `khnsw.core.events.EventType`
 * and `docs/contracts/event.schema.json`.
 */
export const EventTypeSchema = z.enum([
  "run_start",
  "move",
  "cert_fail",
  "enqueue",
  "pop",
  "stale_skip",
  "edge_add",
  "edge_remove",
  "search",
  "metric",
  "run_end",
]);

export type EventType = z.infer<typeof EventTypeSchema>;

/**
 * Zod schema validating kinetic trace events.
 */
export const EventSchema = z.object({
  schema_version: z.literal(1).default(1),
  run_id: z.string(),
  seq: z.number().int().nonnegative(),
  t: z.number().nonnegative(),
  type: EventTypeSchema,
  node: z.number().int().nullable(),
  level: z.number().int().nonnegative().nullable(),
  version: z.number().int().nonnegative().nullable(),
  payload: z.record(z.string(), z.unknown()),
});

export type Event = z.infer<typeof EventSchema>;

