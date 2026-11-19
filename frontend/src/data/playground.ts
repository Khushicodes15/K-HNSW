export type MovementMode = "replacement" | "linear" | "cluster" | "adversarial";

export interface PlaygroundNode {
  id: string;
  index: number;
  layer: number;
  priority: number;
  recallLoss: number;
  repairCost: number;
  x: number;
  y: number;
}

export interface PlaygroundEdge {
  source: string;
  target: string;
  level: number;
  violated?: boolean;
  added?: boolean;
}

export interface Violation {
  nodeId: string;
  targetId: string;
  level: number;
  detectedAt: number;
  failureAt: number;
  detail: string;
}

export interface ScheduledRepair extends PlaygroundNode {
  fitsBudget: boolean;
}

export const playgroundNodes: PlaygroundNode[] = Array.from({ length: 36 }, (_, index) => {
  const angle = index * 2.39996;
  const radius = 0.12 + ((index * 17) % 31) / 100;
  return {
    id: `N-${String(index + 1).padStart(2, "0")}`,
    index,
    layer: index < 2 ? 2 : index < 9 ? 1 : 0,
    priority: Math.max(26, 96 - ((index * 19) % 67)),
    recallLoss: Number((0.18 + ((index * 13) % 49) / 100).toFixed(2)),
    repairCost: 8 + ((index * 7) % 20),
    x: 0.5 + Math.cos(angle) * radius,
    y: 0.5 + Math.sin(angle) * radius * 0.75,
  };
});

export const violations: Violation[] = [
  {
    nodeId: "N-08",
    targetId: "N-12",
    level: 0,
    detectedAt: 1.8,
    failureAt: 5.2,
    detail: "Distance certificate crossed its validity bound after node movement.",
  },
  {
    nodeId: "N-14",
    targetId: "N-19",
    level: 0,
    detectedAt: 3.1,
    failureAt: 7.6,
    detail: "Neighbor ordering is no longer certified for the current vector positions.",
  },
  {
    nodeId: "N-21",
    targetId: "N-25",
    level: 0,
    detectedAt: 4.4,
    failureAt: 9.3,
    detail: "The edge's conservative separation bound has been exceeded.",
  },
  {
    nodeId: "N-29",
    targetId: "N-32",
    level: 0,
    detectedAt: 6.2,
    failureAt: 11.1,
    detail: "Movement invalidated the stored local-neighborhood certificate.",
  },
];

const edgeKey = (source: string, target: string, level: number) =>
  `${[source, target].sort().join(":")}:${level}`;

const edgeMap = new Map<string, PlaygroundEdge>();
const layerNodes = (level: number) => playgroundNodes.filter((node) => node.layer >= level);

for (let level = 0; level <= 2; level += 1) {
  const nodes = layerNodes(level);
  const linksPerNode = level === 0 ? [1, 4, 9] : level === 1 ? [1, 2] : [1];
  nodes.forEach((node, index) => {
    linksPerNode.forEach((step) => {
      const target = nodes[(index + step) % nodes.length];
      if (target.id !== node.id) {
        edgeMap.set(edgeKey(node.id, target.id, level), {
          source: node.id,
          target: target.id,
          level,
        });
      }
    });
  });
}

violations.forEach((violation) => {
  edgeMap.set(edgeKey(violation.nodeId, violation.targetId, violation.level), {
    source: violation.nodeId,
    target: violation.targetId,
    level: violation.level,
    violated: true,
  });
});

export const playgroundEdges = [...edgeMap.values()];

export const rewiringRemovedEdges = playgroundEdges.filter((edge) => edge.violated);
export const rewiringAddedEdges: PlaygroundEdge[] = [
  { source: "N-08", target: "N-10", level: 0, added: true },
  { source: "N-14", target: "N-16", level: 0, added: true },
  { source: "N-21", target: "N-23", level: 0, added: true },
  { source: "N-29", target: "N-31", level: 0, added: true },
];
export const rewiringAffectedNodes = [
  ...new Set(
    [...rewiringRemovedEdges, ...rewiringAddedEdges].flatMap((edge) => [
      edge.source,
      edge.target,
    ]),
  ),
];

export function edgesForRepair(state: "before" | "after"): PlaygroundEdge[] {
  if (state === "before") return playgroundEdges;
  return [
    ...playgroundEdges.filter((edge) => !edge.violated),
    ...rewiringAddedEdges,
  ];
}

export function getMovementOffset(
  node: PlaygroundNode,
  mode: MovementMode,
  time: number,
): [number, number] {
  const step = Math.floor(time / 3);
  switch (mode) {
    case "replacement":
      return node.index % 6 === step % 6
        ? [step % 2 === 0 ? 0.045 : -0.045, step % 2 === 0 ? -0.025 : 0.025]
        : [0, 0];
    case "linear":
      return [time * ((node.index % 2 === 0 ? 1 : -1) * 0.002), time * 0.001];
    case "cluster": {
      const cluster = Math.floor(node.index / 6);
      return [time * (cluster % 2 === 0 ? 0.0017 : -0.0017), time * 0.0007];
    }
    case "adversarial":
      return node.index % 4 === step % 4 ? [0.065, -0.045] : [0, 0];
  }
}

export function createRepairSchedule(budget: number): ScheduledRepair[] {
  let remaining = Math.max(0, budget);
  return [...playgroundNodes]
    .filter((node) => violations.some((violation) => violation.nodeId === node.id))
    .sort((a, b) => b.priority - a.priority)
    .map((node) => {
      const fitsBudget = node.repairCost <= remaining;
      if (fitsBudget) remaining -= node.repairCost;
      return { ...node, fitsBudget };
    });
}

export function movementSeries(mode: MovementMode) {
  return Array.from({ length: 13 }, (_, time) => {
    const movedNodes = playgroundNodes.filter(
      (node) => getMovementOffset(node, mode, time)[0] !== 0 || getMovementOffset(node, mode, time)[1] !== 0,
    );
    const displacement = movedNodes.reduce((sum, node) => {
      const [x, y] = getMovementOffset(node, mode, time);
      return sum + Math.hypot(x, y);
    }, 0);
    return {
      time,
      displacement: Number(displacement.toFixed(3)),
      moved: movedNodes.length,
    };
  });
}
