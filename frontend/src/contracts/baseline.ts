import { z } from "zod";

export const BaselineMethodSchema = z.enum([
  "static_hnsw",
  "native_update",
  "delete_reinsert",
  "full_rebuild",
  "khnsw",
]);

export const BaselineResultSchema = z.object({
  method: BaselineMethodSchema,
  recallAtK: z.number().finite().min(0).max(1).optional(),
  p50LatencyMs: z.number().finite().nonnegative().optional(),
  qps: z.number().finite().nonnegative().optional(),
  memoryMb: z.number().finite().nonnegative().optional(),
});

export const BaselineDatasetSchema = z
  .object({
    schema_version: z.literal(1),
    run_id: z.string().min(1),
    dataset: z.string().min(1),
    k: z.number().int().positive(),
    results: z.array(BaselineResultSchema).min(1),
  })
  .superRefine((dataset, context) => {
    const seen = new Set<string>();
    dataset.results.forEach((result, index) => {
      if (seen.has(result.method)) {
        context.addIssue({
          code: "custom",
          message: `Duplicate result for ${result.method}.`,
          path: ["results", index, "method"],
        });
      }
      seen.add(result.method);
    });
  });

export type BaselineMethod = z.infer<typeof BaselineMethodSchema>;
export type BaselineResult = z.infer<typeof BaselineResultSchema>;
export type BaselineDataset = z.infer<typeof BaselineDatasetSchema>;

export const baselineMethods: { id: BaselineMethod; label: string }[] = [
  { id: "static_hnsw", label: "Static HNSW" },
  { id: "native_update", label: "Native update" },
  { id: "delete_reinsert", label: "Delete & reinsert" },
  { id: "full_rebuild", label: "Full rebuild" },
  { id: "khnsw", label: "K-HNSW" },
];
