import { describe, expect, it } from "vitest";
import {
  createRepairSchedule,
  edgesForRepair,
  getMovementOffset,
  playgroundNodes,
  rewiringAddedEdges,
  rewiringRemovedEdges,
} from "./playground";

describe("playground demo models", () => {
  it("keeps every scheduled repair within the available budget", () => {
    const budget = 32;
    const schedule = createRepairSchedule(budget);
    const allocated = schedule
      .filter((repair) => repair.fitsBudget)
      .reduce((total, repair) => total + repair.repairCost, 0);

    expect(allocated).toBeLessThanOrEqual(budget);
    expect(schedule.map((repair) => repair.priority)).toEqual(
      [...schedule.map((repair) => repair.priority)].sort((a, b) => b - a),
    );
    expect(createRepairSchedule(0).every((repair) => !repair.fitsBudget)).toBe(true);
    expect(createRepairSchedule(Number.MAX_SAFE_INTEGER).every((repair) => repair.fitsBudget)).toBe(true);
  });

  it("applies discrete replacements at step boundaries", () => {
    const node = playgroundNodes[0];

    expect(getMovementOffset(node, "replacement", 2.9)).toEqual([0.045, -0.025]);
    expect(getMovementOffset(node, "replacement", 3)).toEqual([0, 0]);
  });

  it("moves nodes in opposite directions during the linear drift model", () => {
    const first = getMovementOffset(playgroundNodes[0], "linear", 5);
    const second = getMovementOffset(playgroundNodes[1], "linear", 5);

    expect(first[0]).toBeGreaterThan(0);
    expect(second[0]).toBeLessThan(0);
    expect(first[1]).toBe(second[1]);
  });

  it("keeps clustered nodes together and changes the adversarial target by step", () => {
    const first = getMovementOffset(playgroundNodes[0], "cluster", 3);
    const sameCluster = getMovementOffset(playgroundNodes[1], "cluster", 3);

    expect(first).toEqual(sameCluster);
    expect(getMovementOffset(playgroundNodes[0], "adversarial", 0)).not.toEqual([0, 0]);
    expect(getMovementOffset(playgroundNodes[0], "adversarial", 3)).toEqual([0, 0]);
    expect(getMovementOffset(playgroundNodes[1], "adversarial", 3)).not.toEqual([0, 0]);
  });

  it("removes invalid links and adds replacements in the after-repair graph", () => {
    const before = edgesForRepair("before");
    const after = edgesForRepair("after");

    expect(before.filter((edge) => edge.violated)).toHaveLength(rewiringRemovedEdges.length);
    expect(after.some((edge) => edge.violated)).toBe(false);
    expect(after.filter((edge) => edge.added)).toHaveLength(rewiringAddedEdges.length);
  });
});
