import { EventSchema, type Event } from "../contracts/event";

export function parseEventTrace(text: string): Event[] {
  const events: Event[] = [];
  let runId: string | null = null;
  let previousSequence = -1;

  for (const [index, rawLine] of text.split(/\r?\n/).entries()) {
    const line = rawLine.trim();
    if (!line) continue;

    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Invalid JSON.";
      throw new Error(`Invalid JSON on line ${index + 1}: ${message}`);
    }

    const parsed = EventSchema.safeParse(value);
    if (!parsed.success) {
      const message = parsed.error.issues[0]?.message ?? "Event does not match the trace schema.";
      throw new Error(`Invalid event on line ${index + 1}: ${message}`);
    }

    if (runId !== null && parsed.data.run_id !== runId) {
      throw new Error(`Trace contains more than one run_id (line ${index + 1}).`);
    }
    if (parsed.data.seq <= previousSequence) {
      throw new Error(`Event sequence must increase at line ${index + 1}.`);
    }

    runId = parsed.data.run_id;
    previousSequence = parsed.data.seq;
    events.push(parsed.data);
  }

  if (events.length === 0) throw new Error("The selected trace contains no events.");
  return events;
}
