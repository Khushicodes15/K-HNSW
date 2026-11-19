import { describe, expect, it } from "vitest";
import { BaselineDatasetSchema } from "./baseline";

describe("BaselineDatasetSchema", () => {
  it("accepts a measured comparison with unavailable metrics omitted", () => {
    const result = BaselineDatasetSchema.safeParse({
      schema_version: 1,
      run_id: "run-1",
      dataset: "local benchmark",
      k: 10,
      results: [{ method: "khnsw", recallAtK: 0.91 }],
    });

    expect(result.success).toBe(true);
  });

  it("rejects recall outside its documented fraction range and duplicate methods", () => {
    const invalid = BaselineDatasetSchema.safeParse({
      schema_version: 1,
      run_id: "run-1",
      dataset: "local benchmark",
      k: 10,
      results: [
        { method: "khnsw", recallAtK: 1.2 },
        { method: "khnsw", recallAtK: 0.9 },
      ],
    });

    expect(invalid.success).toBe(false);
  });
});
