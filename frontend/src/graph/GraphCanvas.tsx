import { useCallback, useEffect, useMemo, useRef } from "react";
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  select,
  zoom,
  zoomIdentity,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
  type ZoomBehavior,
  type ZoomTransform,
} from "d3";
import {
  edgesForRepair,
  getMovementOffset,
  playgroundEdges,
  playgroundNodes,
  rewiringAffectedNodes,
  type MovementMode,
} from "../data/playground";

type GraphState = "current" | "before" | "after";

interface GraphNode extends SimulationNodeDatum {
  id: string;
  index: number;
  layer: number;
  x: number;
  y: number;
}

interface GraphCanvasProps {
  layer: number;
  selectedNode: string;
  movementMode: MovementMode;
  time: number;
  highlightViolations?: boolean;
  graphState?: GraphState;
  onSelectNode: (id: string) => void;
}

export function GraphCanvas({
  layer,
  selectedNode,
  movementMode,
  time,
  highlightViolations = true,
  graphState = "current",
  onSelectNode,
}: GraphCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const transformRef = useRef<ZoomTransform>(zoomIdentity);
  const zoomBehaviorRef = useRef<ZoomBehavior<HTMLCanvasElement, unknown> | null>(null);
  const drawRef = useRef<() => void>(() => undefined);
  const nodes = useMemo(
    () => playgroundNodes.map((node) => ({ ...node })),
    [],
  );

  const drawGraph = useCallback(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    const bounds = canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = bounds.width * ratio;
    canvas.height = bounds.height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);
    context.save();
    context.translate(transformRef.current.x, transformRef.current.y);
    context.scale(transformRef.current.k, transformRef.current.k);

    const positions = new Map(
      nodes.map((node) => {
        const [dx, dy] = getMovementOffset(node, movementMode, time);
        return [node.id, { x: (node.x + dx) * bounds.width, y: (node.y + dy) * bounds.height }];
      }),
    );
    const graphEdges = graphState === "current" ? playgroundEdges : edgesForRepair(graphState);
    const visibleEdges = graphEdges.filter((edge) => edge.level === layer);

    for (const edge of visibleEdges) {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) continue;
      const isSelected = edge.source === selectedNode || edge.target === selectedNode;
      const isBroken = highlightViolations && edge.violated;
      const isAdded = graphState === "after" && edge.added;
      context.beginPath();
      context.moveTo(source.x, source.y);
      context.lineTo(target.x, target.y);
      context.strokeStyle = isBroken
        ? "rgba(185, 28, 58, 0.88)"
        : isAdded
          ? "rgba(21, 128, 86, 0.9)"
        : isSelected
          ? "rgba(124, 58, 237, 0.82)"
          : "rgba(100, 116, 139, 0.34)";
      context.lineWidth = isBroken || isAdded ? 1.5 : isSelected ? 1.2 : 0.8;
      context.setLineDash(isBroken ? [4, 4] : []);
      context.stroke();
    }

    for (const node of nodes) {
      if (node.layer < layer) continue;
      const position = positions.get(node.id);
      if (!position) continue;
      const isSelected = node.id === selectedNode;
      const isViolated =
        highlightViolations &&
        graphEdges.some(
          (edge) =>
            edge.violated && (edge.source === node.id || edge.target === node.id),
        );
      const isAffected =
        graphState !== "current" && rewiringAffectedNodes.includes(node.id);
      context.beginPath();
      context.arc(position.x, position.y, isSelected ? 6 : isViolated || isAffected ? 4.8 : 3.6, 0, Math.PI * 2);
      context.fillStyle = isSelected
        ? "#7c3aed"
        : isViolated
          ? "#c24155"
          : graphState === "after" && isAffected
            ? "#16845b"
            : "#718096";
      context.fill();
      if (isSelected || isAffected) {
        context.beginPath();
        context.arc(position.x, position.y, isSelected ? 10 : 8, 0, Math.PI * 2);
        context.strokeStyle = isSelected
          ? "rgba(124, 58, 237, .3)"
          : graphState === "after"
            ? "rgba(22, 132, 91, .3)"
            : "rgba(194, 65, 85, .3)";
        context.lineWidth = 1;
        context.stroke();
      }
    }

    context.restore();
  }, [graphState, highlightViolations, layer, movementMode, nodes, selectedNode, time]);

  useEffect(() => {
    drawRef.current = drawGraph;
    drawGraph();
  }, [drawGraph]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const graphEdges = graphState === "current" ? playgroundEdges : edgesForRepair(graphState);
    const links: SimulationLinkDatum<GraphNode>[] = graphEdges
      .filter((edge) => edge.level === layer)
      .map((edge) => ({ source: edge.source, target: edge.target }));

    forceSimulation<GraphNode>(nodes)
      .force(
        "link",
        forceLink<GraphNode, SimulationLinkDatum<GraphNode>>(links)
          .id((node) => node.id)
          .distance(0.105)
          .strength(0.32),
      )
      .force("charge", forceManyBody<GraphNode>().strength(-0.035))
      .force("center", forceCenter(0.5, 0.5))
      .force("collide", forceCollide<GraphNode>(0.035))
      .stop()
      .tick(180);

    const behavior = zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.45, 4])
      .clickDistance(3)
      .on("zoom", (event) => {
        transformRef.current = event.transform;
        drawRef.current();
      });
    zoomBehaviorRef.current = behavior;
    select(canvas).call(behavior).on("dblclick.zoom", null);

    const observer = new ResizeObserver(() => drawRef.current());
    observer.observe(canvas);
    drawRef.current();

    return () => {
      observer.disconnect();
      select(canvas).on(".zoom", null);
      zoomBehaviorRef.current = null;
    };
  }, [graphState, layer, nodes]);

  const zoomBy = (scale: number) => {
    const canvas = canvasRef.current;
    const behavior = zoomBehaviorRef.current;
    if (canvas && behavior) select(canvas).transition().duration(140).call(behavior.scaleBy, scale);
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const bounds = canvas.getBoundingClientRect();
    const [x, y] = transformRef.current.invert([event.nativeEvent.offsetX, event.nativeEvent.offsetY]);
    const nearest = nodes
      .filter((node) => node.layer >= layer)
      .map((node) => {
        const [dx, dy] = getMovementOffset(node, movementMode, time);
        const nx = (node.x + dx) * bounds.width;
        const ny = (node.y + dy) * bounds.height;
        return { node, distance: Math.hypot(nx - x, ny - y) };
      })
      .sort((a, b) => a.distance - b.distance)[0];
    if (nearest && nearest.distance < 15 / transformRef.current.k) {
      onSelectNode(nearest.node.id);
    }
  };

  return (
    <div className="graph-canvas-wrap">
      <canvas
        ref={canvasRef}
        className="graph-canvas"
        aria-label={`Interactive HNSW graph, layer ${layer}. Drag to pan, scroll to zoom, click a node to inspect.`}
        onClick={handleCanvasClick}
      />
      <div className="canvas-controls" aria-label="Graph zoom controls">
        <button type="button" onClick={() => zoomBy(1.25)} aria-label="Zoom in">
          +
        </button>
        <button type="button" onClick={() => zoomBy(0.8)} aria-label="Zoom out">
          −
        </button>
        <button
          type="button"
          onClick={() => {
            const canvas = canvasRef.current;
            const behavior = zoomBehaviorRef.current;
            if (canvas && behavior) select(canvas).transition().duration(140).call(behavior.transform, zoomIdentity);
          }}
          aria-label="Reset graph view"
        >
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M3 7a5 5 0 1 1 1.2 3.2M3 3v4h4" />
          </svg>
        </button>
      </div>
      <span className="canvas-hint">Scroll to zoom · drag to pan · click a node</span>
    </div>
  );
}
