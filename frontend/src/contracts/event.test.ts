import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { EventSchema } from "./event";

describe("EventSchema contract validation", () => {
  it("validates every line of sample_run.jsonl against Zod schema", () => {
    const sampleRunPath = path.resolve(__dirname, "../../../docs/contracts/examples/sample_run.jsonl");
    expect(fs.existsSync(sampleRunPath)).toBe(true);

    const fileContent = fs.readFileSync(sampleRunPath, "utf-8");
    const lines = fileContent.split("\n").filter((l) => l.trim().length > 0);

    expect(lines.length).toBeGreaterThanOrEqual(6);

    for (const [index, line] of lines.entries()) {
      const raw = JSON.parse(line);
      const parsed = EventSchema.parse(raw);

      expect(parsed.schema_version).toBe(1);
      expect(parsed.payload.example).toBe(true);
      expect(parsed.seq).toBe(index);
    }
  });

  it("rejects invalid event objects", () => {
    const invalidEvent = {
      schema_version: 2, // invalid version
      run_id: "test",
      seq: -1, // negative
      t: -0.5,
      type: "non_existent_type",
      node: "not_a_number",
    };

    const result = EventSchema.safeParse(invalidEvent);
    expect(result.success).toBe(false);
  });
});

