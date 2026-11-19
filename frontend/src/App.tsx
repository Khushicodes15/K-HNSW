import {
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { GraphCanvas } from "./graph/GraphCanvas";
import {
  createRepairSchedule,
  edgesForRepair,
  movementSeries,
  playgroundEdges,
  playgroundNodes,
  rewiringAddedEdges,
  rewiringAffectedNodes,
  rewiringRemovedEdges,
  violations,
  type MovementMode,
} from "./data/playground";
import {
  baselineMethods,
  BaselineDatasetSchema,
  type BaselineDataset,
} from "./contracts/baseline";
import type { Event } from "./contracts/event";
import { deriveLiveMetrics } from "./replay/metrics";
import "./App.css";

const ChartPanel = lazy(() => import("./dashboard/ChartPanel"));

type View =
  | "graph"
  | "movement"
  | "violations"
  | "priority"
  | "rewiring"
  | "baseline"
  | "metrics";

const viewTitles: Record<View, { eyebrow: string; title: string; description: string }> = {
  graph: {
    eyebrow: "GRAPH EXPLORER",
    title: "Graph playground",
    description: "Explore topology across layers and inspect a node's neighborhood.",
  },
  movement: {
    eyebrow: "KINETIC REPLAY",
    title: "Movement simulator",
    description: "Preview how different vector trajectories change the graph over time.",
  },
  violations: {
    eyebrow: "CERTIFICATE MONITOR",
    title: "Violation view",
    description: "Inspect invalidated certificates and their predicted failure windows.",
  },
  priority: {
    eyebrow: "REPAIR SCHEDULER",
    title: "Priority view",
    description: "Rank repair candidates and see which tasks fit the available budget.",
  },
  rewiring: {
    eyebrow: "TOPOLOGY DIFF",
    title: "Rewiring view",
    description: "Compare graph structure before and after a repair pass.",
  },
  baseline: {
    eyebrow: "BENCHMARKS",
    title: "Baseline vs K-HNSW",
    description: "Compare measured results from one benchmark run across update strategies.",
  },
  metrics: {
    eyebrow: "ENGINE TELEMETRY",
    title: "Live metrics",
    description: "Replay an event trace and inspect its query, repair, and connectivity metrics.",
  },
};

const movementOptions: { id: MovementMode; title: string; note: string }[] = [
  { id: "replacement", title: "Discrete replacement", note: "Step changes" },
  { id: "linear", title: "Linear drift", note: "Constant velocity" },
  { id: "cluster", title: "Cluster drift", note: "Group movement" },
  { id: "adversarial", title: "Adversarial", note: "Stress trajectory" },
];

const numberLabel = (value: number) => String(value).padStart(2, "0");

function NavigationIcon({ view }: { view: View }) {
  if (view === "graph") {
    return (
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <circle cx="4" cy="10" r="2" />
        <circle cx="15.5" cy="4" r="2" />
        <circle cx="15.5" cy="16" r="2" />
        <path d="m5.8 9 7.8-4M5.8 11l7.8 4" />
      </svg>
    );
  }
  if (view === "movement") {
    return (
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <path d="M2.5 10h3l2.2-5 4.1 10 2.2-5h3.5" />
      </svg>
    );
  }
  if (view === "violations") {
    return (
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <path d="M10 2.5 18 17H2L10 2.5Z" />
        <path d="M10 7v4.5M10 14.5v.1" />
      </svg>
    );
  }
  if (view === "priority") {
    return (
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <path d="M4 16.5V10M10 16.5V4M16 16.5V7" />
        <path d="M2 17.5h16" />
      </svg>
    );
  }
  if (view === "rewiring") {
    return (
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <circle cx="4" cy="5" r="2" />
        <circle cx="16" cy="5" r="2" />
        <circle cx="10" cy="16" r="2" />
        <path d="M6 5h3m2 0h3M5.2 6.7l3.6 7.5m2.4 0 3.6-7.5" />
      </svg>
    );
  }
  if (view === "baseline") {
    return (
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <path d="M3 16.5h14M5 13V8m5 5V4m5 9V6" />
        <path d="M3.5 4.5h3M8.5 2h3M13.5 4h3" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <path d="M4 16.5V10M10 16.5V4M16 16.5V7" />
      <path d="M2 17.5h16" />
    </svg>
  );
}

function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 36 36" aria-hidden="true">
      <path d="m18 3 14 8v14l-14 8-14-8V11l14-8Z" />
      <path d="m18 10 8 4.5v7L18 26l-8-4.5v-7l8-4.5Z" />
      <circle cx="18" cy="18" r="2" />
    </svg>
  );
}

function MetricCard({
  label,
  value,
  caption,
  icon,
  tone,
}: {
  label: string;
  value: string;
  caption: string;
  icon: string;
  tone: string;
}) {
  return (
    <article className="metric-card">
      <div className={`metric-icon ${tone}`}>{icon}</div>
      <div className="metric-copy">
        <span className="metric-label">{label}</span>
        <strong>{value}</strong>
        <span className="metric-caption">{caption}</span>
      </div>
    </article>
  );
}

function GraphToolbar({
  layer,
  onChangeLayer,
  graphState,
}: {
  layer: number;
  onChangeLayer: (layer: number) => void;
  graphState: "current" | "before" | "after";
}) {
  return (
    <div className="graph-toolbar">
      <div className="layer-tabs" aria-label="Graph layer">
        {[0, 1, 2].map((level) => (
          <button
            type="button"
            className={layer === level ? "layer-tab active" : "layer-tab"}
            key={level}
            onClick={() => onChangeLayer(level)}
            aria-pressed={layer === level}
          >
            Layer {level}
          </button>
        ))}
      </div>
      <div className="graph-legend">
        {graphState === "before" ? (
          <>
            <span><i className="legend-dot normal" /> Node</span>
            <span><i className="diff-line removed-line" /> Removed</span>
          </>
        ) : graphState === "after" ? (
          <>
            <span><i className="legend-dot normal" /> Node</span>
            <span><i className="diff-line added-line" /> Added</span>
          </>
        ) : (
          <>
            <span><i className="legend-dot normal" /> Node</span>
            <span><i className="legend-dot selected" /> Selected</span>
            <span><i className="legend-dot warning" /> Violation</span>
          </>
        )}
      </div>
    </div>
  );
}

function GraphPanel({
  layer,
  onChangeLayer,
  selectedNode,
  onSelectNode,
  movementMode,
  time,
  highlightViolations = true,
  graphState = "current",
  eyebrow = "TOPOLOGY",
  title = "Live graph",
}: {
  layer: number;
  onChangeLayer: (layer: number) => void;
  selectedNode: string;
  onSelectNode: (id: string) => void;
  movementMode: MovementMode;
  time: number;
  highlightViolations?: boolean;
  graphState?: "current" | "before" | "after";
  eyebrow?: string;
  title?: string;
}) {
  const graphEdges = graphState === "current" ? playgroundEdges : edgesForRepair(graphState);
  const layerEdges = graphEdges.filter((edge) => edge.level === layer).length;
  const layerNodes = playgroundNodes.filter((node) => node.layer >= layer).length;
  return (
    <section className={`panel graph-panel graph-panel-${graphState}`}>
      <div className="panel-heading graph-heading">
        <div>
          <span className="eyebrow">{eyebrow}</span>
          <h2>{title}</h2>
        </div>
        <div className="graph-counts">
          <span><strong>{numberLabel(layerNodes)}</strong> nodes</span>
          <span className="count-divider" />
          <span><strong>{numberLabel(layerEdges)}</strong> edges</span>
        </div>
      </div>
      <GraphToolbar layer={layer} onChangeLayer={onChangeLayer} graphState={graphState} />
      <GraphCanvas
        layer={layer}
        selectedNode={selectedNode}
        onSelectNode={onSelectNode}
        movementMode={movementMode}
        time={time}
        highlightViolations={highlightViolations}
        graphState={graphState}
      />
      <div className="graph-footer">
        <span><i className="live-dot" /> Demo trace · 36 indexed nodes</span>
        <span>Drag, scroll, or select a node</span>
      </div>
    </section>
  );
}

function NodeInspector({ selectedNode, layer }: { selectedNode: string; layer: number }) {
  const node = playgroundNodes.find((item) => item.id === selectedNode) ?? playgroundNodes[0];
  const edges = playgroundEdges.filter(
    (edge) => edge.level === layer && (edge.source === node.id || edge.target === node.id),
  );
  const neighborIds = [...new Set(edges.map((edge) => edge.source === node.id ? edge.target : edge.source))];
  const isFlagged = violations.some((violation) => violation.nodeId === node.id);
  return (
    <section className="panel inspector-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">NODE INSPECTOR</span>
          <h2>{node.id}</h2>
        </div>
        <span className={isFlagged ? "state-pill warning-pill" : "state-pill healthy-pill"}>
          <i /> {isFlagged ? "Flagged" : "Healthy"}
        </span>
      </div>
      <div className="node-summary">
        <div className="node-avatar">{node.id.slice(-2)}</div>
        <div>
          <strong>Layer {node.layer} member</strong>
          <span>Selected from layer {layer}</span>
        </div>
      </div>
      <div className="inspector-stats">
        <div><span>Priority score</span><strong>{node.priority}<small> / 100</small></strong></div>
        <div><span>Neighbors here</span><strong>{neighborIds.length}<small> edges</small></strong></div>
      </div>
      <div className="section-rule" />
      <div className="subheading-row">
        <h3>Connections</h3>
        <span className="subtle-count">{neighborIds.length} neighbors</span>
      </div>
      {neighborIds.length ? (
        <div className="neighbor-list">
          {neighborIds.slice(0, 6).map((id) => {
            const edge = edges.find((item) => item.source === id || item.target === id);
            return (
              <div className="neighbor-row" key={id}>
                <span className={`neighbor-node ${edge?.violated ? "has-warning" : ""}`} />
                <span>{id}</span>
                <span className="neighbor-level">L{layer}</span>
                <span className={edge?.violated ? "edge-status invalid" : "edge-status"}>
                  {edge?.violated ? "invalid" : "certified"}
                </span>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="empty-note">No visible connections at this layer.</p>
      )}
      <div className="inspector-callout">
        <span className="callout-glyph">↗</span>
        <p>Node scores and topology shown here are illustrative demo data.</p>
      </div>
    </section>
  );
}

function GraphView({
  layer,
  setLayer,
  selectedNode,
  setSelectedNode,
  movementMode,
  time,
}: {
  layer: number;
  setLayer: (layer: number) => void;
  selectedNode: string;
  setSelectedNode: (id: string) => void;
  movementMode: MovementMode;
  time: number;
}) {
  return (
    <>
      <div className="metrics-grid">
        <MetricCard label="Graph nodes" value="36" caption="Demo topology" icon="◌" tone="violet" />
        <MetricCard label="Visible edges" value={String(playgroundEdges.filter((edge) => edge.level === layer).length)} caption={`Layer ${layer} connections`} icon="⌘" tone="blue" />
        <MetricCard label="Cert violations" value="04" caption="Illustrative replay" icon="!" tone="rose" />
        <MetricCard label="Hierarchy" value="03" caption="Active graph layers" icon="▱" tone="mint" />
      </div>
      <div className="workspace-grid graph-workspace">
        <GraphPanel
          layer={layer}
          onChangeLayer={setLayer}
          selectedNode={selectedNode}
          onSelectNode={setSelectedNode}
          movementMode={movementMode}
          time={time}
        />
        <NodeInspector selectedNode={selectedNode} layer={layer} />
      </div>
      <div className="below-graph-note">
        <span className="note-icon">i</span>
        <p>Illustrative topology for interaction only. Connect a replay trace to inspect a real index.</p>
      </div>
    </>
  );
}

function MovementView({
  layer,
  setLayer,
  selectedNode,
  setSelectedNode,
  movementMode,
  setMovementMode,
  time,
  setTime,
  playing,
  setPlaying,
}: {
  layer: number;
  setLayer: (layer: number) => void;
  selectedNode: string;
  setSelectedNode: (id: string) => void;
  movementMode: MovementMode;
  setMovementMode: (mode: MovementMode) => void;
  time: number;
  setTime: (time: number) => void;
  playing: boolean;
  setPlaying: (playing: boolean) => void;
}) {
  const series = useMemo(() => movementSeries(movementMode), [movementMode]);
  const current = series[Math.min(Math.round(time), series.length - 1)];
  return (
    <>
      <section className="panel movement-controls">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">MOVEMENT MODEL</span>
            <h2>Choose a trajectory</h2>
          </div>
          <span className="demo-tag">LOCAL DEMO</span>
        </div>
        <div className="movement-options">
          {movementOptions.map((option, index) => (
            <button
              type="button"
              key={option.id}
              className={`movement-option ${movementMode === option.id ? "active" : ""}`}
              onClick={() => setMovementMode(option.id)}
              aria-pressed={movementMode === option.id}
            >
              <span className={`movement-option-icon model-${index}`}>{["↻", "↗", "◉", "⌁"][index]}</span>
              <span><strong>{option.title}</strong><small>{option.note}</small></span>
              <i className="radio-dot" />
            </button>
          ))}
        </div>
        <div className="timeline">
          <button
            type="button"
            className="play-button"
            aria-label={playing ? "Pause simulation" : "Play simulation"}
            onClick={() => setPlaying(!playing)}
          >
            {playing ? "Ⅱ" : "▶"}
          </button>
          <button
            type="button"
            className="step-button"
            aria-label="Advance simulation one step"
            onClick={() => setTime(time >= 12 ? 0 : time + 1)}
          >
            ↦
          </button>
          <div className="timeline-range">
            <div className="timeline-labels"><span>t = {time.toFixed(1)}s</span><span>12.0s</span></div>
            <input
              aria-label="Simulation time"
              type="range"
              min="0"
              max="12"
              step="0.1"
              value={time}
              onChange={(event) => setTime(Number(event.target.value))}
            />
          </div>
          <span className="timeline-readout"><strong>{numberLabel(current.moved)}</strong><small>moved</small></span>
        </div>
      </section>
      <div className="workspace-grid movement-workspace">
        <GraphPanel
          layer={layer}
          onChangeLayer={setLayer}
          selectedNode={selectedNode}
          onSelectNode={setSelectedNode}
          movementMode={movementMode}
          time={time}
          eyebrow="MOVING TOPOLOGY"
          title="Graph response"
        />
        <section className="panel movement-chart-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">TRAJECTORY PROFILE</span>
              <h2>Aggregate displacement</h2>
            </div>
            <span className="chart-unit">model units</span>
          </div>
          <div className="chart-wrap movement-chart">
            <Suspense fallback={<div className="chart-loading">Loading chart…</div>}>
              <ChartPanel type="movement" data={series} />
            </Suspense>
          </div>
          <div className="chart-legend-row"><span><i className="chart-line" /> Aggregate displacement</span><span className="chart-demo-label">Illustrative model</span></div>
          <div className="section-rule" />
          <div className="movement-summary">
            <span>Active model</span>
            <strong>{movementOptions.find((option) => option.id === movementMode)?.title}</strong>
            <p>Positions are projected from the selected demo trajectory; no backend index is connected.</p>
          </div>
        </section>
      </div>
    </>
  );
}

function ViolationView({
  layer,
  setLayer,
  selectedNode,
  setSelectedNode,
  movementMode,
  time,
}: {
  layer: number;
  setLayer: (layer: number) => void;
  selectedNode: string;
  setSelectedNode: (id: string) => void;
  movementMode: MovementMode;
  time: number;
}) {
  const currentViolation = violations.find((item) => item.nodeId === selectedNode);
  const activeCount = violations.filter((item) => time >= item.detectedAt).length;
  return (
    <>
      <div className="violation-banner">
        <div className="banner-symbol">!</div>
        <div><strong>{numberLabel(activeCount)} certificates invalidated</strong><span>at t = {time.toFixed(1)}s in this illustrative replay</span></div>
        <span className="banner-context">DEMO TRACE</span>
      </div>
      <div className="workspace-grid violation-workspace">
        <GraphPanel
          layer={layer}
          onChangeLayer={setLayer}
          selectedNode={selectedNode}
          onSelectNode={setSelectedNode}
          movementMode={movementMode}
          time={time}
          eyebrow="AFFECTED TOPOLOGY"
          title="Problem edges"
        />
        <section className="panel violations-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">CERTIFICATE EVENTS</span>
              <h2>Needs attention <span className="title-count">{numberLabel(violations.length)}</span></h2>
            </div>
            <span className="sort-label">BY FAILURE TIME <span>⌄</span></span>
          </div>
          <div className="violation-list">
            {violations.map((violation, index) => {
              const isActive = time >= violation.detectedAt;
              const selected = currentViolation?.nodeId === violation.nodeId;
              return (
                <button
                  type="button"
                  key={violation.nodeId}
                  className={`violation-item ${selected ? "selected" : ""}`}
                  onClick={() => setSelectedNode(violation.nodeId)}
                >
                  <span className={isActive ? "violation-number active" : "violation-number"}>{numberLabel(index + 1)}</span>
                  <span className="violation-main">
                    <span className="violation-edge">{violation.nodeId}<b>→</b>{violation.targetId}<small>L{violation.level}</small></span>
                    <span className="violation-status">{isActive ? "Certificate violated" : `Predicted at ${violation.detectedAt.toFixed(1)}s`}</span>
                  </span>
                  <span className="failure-time">
                    <strong>{Math.max(0, violation.failureAt - time).toFixed(1)}s</strong>
                    <small>to failure</small>
                  </span>
                </button>
              );
            })}
          </div>
          <div className="section-rule" />
          {currentViolation ? (
            <>
              <div className="detail-heading"><span className="eyebrow">SELECTED DETAIL</span><span className="repair-needed"><i /> REPAIR NEEDED</span></div>
              <p className="violation-detail">{currentViolation.detail}</p>
              <div className="detail-facts">
                <div><span>Source node</span><strong>{currentViolation.nodeId}</strong></div>
                <div><span>Affected edge</span><strong>{currentViolation.nodeId} → {currentViolation.targetId}</strong></div>
                <div><span>Predicted failure</span><strong>t = {currentViolation.failureAt.toFixed(1)}s</strong></div>
              </div>
              <p className="small-disclaimer">Failure times are illustrative values from the local demo trace.</p>
            </>
          ) : (
            <p className="empty-note">Select a flagged node or certificate event to inspect its details.</p>
          )}
        </section>
      </div>
    </>
  );
}

function RewiringView({
  layer,
  setLayer,
  selectedNode,
  setSelectedNode,
  movementMode,
  time,
}: {
  layer: number;
  setLayer: (layer: number) => void;
  selectedNode: string;
  setSelectedNode: (id: string) => void;
  movementMode: MovementMode;
  time: number;
}) {
  const beforeEdges = edgesForRepair("before").filter((edge) => edge.level === layer);
  const afterEdges = edgesForRepair("after").filter((edge) => edge.level === layer);

  return (
    <>
      <div className="rewire-summary">
        <div><span className="change-marker removed-marker">−</span><strong>{rewiringRemovedEdges.length}</strong><span>removed edges</span></div>
        <div><span className="change-marker added-marker">+</span><strong>{rewiringAddedEdges.length}</strong><span>added edges</span></div>
        <div><span className="change-marker affected-marker">◎</span><strong>{rewiringAffectedNodes.length}</strong><span>affected nodes</span></div>
        <span className="demo-tag">ILLUSTRATIVE REPAIR</span>
      </div>
      <div className="rewire-graphs">
        <GraphPanel
          layer={layer}
          onChangeLayer={setLayer}
          selectedNode={selectedNode}
          onSelectNode={setSelectedNode}
          movementMode={movementMode}
          time={time}
          graphState="before"
          eyebrow="BEFORE REPAIR"
          title="Invalid edges"
        />
        <GraphPanel
          layer={layer}
          onChangeLayer={setLayer}
          selectedNode={selectedNode}
          onSelectNode={setSelectedNode}
          movementMode={movementMode}
          time={time}
          graphState="after"
          eyebrow="AFTER REPAIR"
          title="Rewired graph"
        />
      </div>
      <section className="panel rewire-details">
        <div className="panel-heading">
          <div><span className="eyebrow">REPAIR DIFF</span><h2>Topology changes</h2></div>
          <span className="layer-diff-label">LAYER {layer} · {beforeEdges.length} → {afterEdges.length} edges</span>
        </div>
        <div className="rewire-lists">
          <div className="rewire-list">
            <h3><i className="diff-line removed-line" /> Removed</h3>
            {rewiringRemovedEdges.map((edge) => (
              <button
                type="button"
                key={`${edge.source}-${edge.target}`}
                className="diff-row"
                onClick={() => setSelectedNode(edge.source)}
              >
                <span>{edge.source}<b>→</b>{edge.target}</span><small>L{edge.level}</small>
              </button>
            ))}
          </div>
          <div className="rewire-list">
            <h3><i className="diff-line added-line" /> Added</h3>
            {rewiringAddedEdges.map((edge) => (
              <button
                type="button"
                key={`${edge.source}-${edge.target}`}
                className="diff-row"
                onClick={() => setSelectedNode(edge.source)}
              >
                <span>{edge.source}<b>→</b>{edge.target}</span><small>L{edge.level}</small>
              </button>
            ))}
          </div>
          <div className="rewire-list affected-list">
            <h3><i className="diff-line affected-line" /> Affected nodes</h3>
            <div className="affected-chips">
              {rewiringAffectedNodes.map((nodeId) => (
                <button
                  type="button"
                  className={selectedNode === nodeId ? "affected-chip selected" : "affected-chip"}
                  key={nodeId}
                  onClick={() => setSelectedNode(nodeId)}
                >
                  {nodeId}
                </button>
              ))}
            </div>
          </div>
        </div>
        <p className="small-disclaimer">The before/after topology is a local illustrative repair diff; it is not emitted by a live repair engine.</p>
      </section>
    </>
  );
}

type BaselineMetricKey = "recallAtK" | "p50LatencyMs" | "qps" | "memoryMb";

const baselineMetricLabels: Record<BaselineMetricKey, string> = {
  recallAtK: "Recall@k",
  p50LatencyMs: "p50 latency",
  qps: "Queries per second",
  memoryMb: "Memory",
};

function formatBaselineMetric(
  value: number | undefined,
  metric: BaselineMetricKey,
): string {
  if (value === undefined) return "—";
  if (metric === "recallAtK") return `${(value * 100).toFixed(2)}%`;
  if (metric === "p50LatencyMs") return `${value.toFixed(3)} ms`;
  if (metric === "qps") return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return `${value.toFixed(2)} MB`;
}

function BaselineView({
  dataset,
  setDataset,
}: {
  dataset: BaselineDataset | null;
  setDataset: (dataset: BaselineDataset | null) => void;
}) {
  const [metric, setMetric] = useState<BaselineMetricKey>("recallAtK");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) return;
    setError("");
    setLoading(true);
    try {
      const content = await file.text();
      const raw: unknown = JSON.parse(content);
      const result = BaselineDatasetSchema.safeParse(raw);
      if (!result.success) {
        setError(`Invalid benchmark JSON: ${result.error.issues[0]?.message ?? "Schema mismatch."}`);
        return;
      }
      setDataset(result.data);
    } catch (caught) {
      setError(caught instanceof Error ? `Could not read benchmark JSON: ${caught.message}` : "Could not read benchmark JSON.");
    } finally {
      setLoading(false);
      input.value = "";
    }
  };

  const results = new Map(
    (dataset?.results ?? []).map((result) => [result.method, result]),
  );
  const chartData = baselineMethods.flatMap((method) => {
    const result = results.get(method.id);
    const rawValue = result?.[metric];
    if (rawValue === undefined) return [];
    return [{
      label: method.label,
      value: metric === "recallAtK" ? rawValue * 100 : rawValue,
    }];
  });
  const chartUnit = metric === "recallAtK" ? "%" : metric === "p50LatencyMs" ? "ms" : metric === "qps" ? "qps" : "MB";

  return (
    <>
      <section className="panel baseline-source">
        <div className="baseline-source-copy">
          <span className="eyebrow">BENCHMARK SOURCE</span>
          <h2>{dataset ? dataset.dataset : "No benchmark loaded"}</h2>
          <p>
            {dataset
              ? `Run ${dataset.run_id} · Recall@${dataset.k} · Values shown exactly as supplied.`
              : "Import one run with comparable measurements. Missing metrics remain blank; no demo results are substituted."}
          </p>
        </div>
        <label className="upload-button">
          <input type="file" accept=".json,application/json" aria-label="Upload benchmark results JSON" onChange={handleUpload} />
          <span>{loading ? "Reading…" : "↑  Import results JSON"}</span>
        </label>
      </section>
      {error && <div className="import-error" role="alert">{error}</div>}
      <div className="baseline-layout">
        <section className="panel baseline-table-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">METHOD COMPARISON</span><h2>Measured results</h2></div>
            {dataset && <span className="source-tag">RUN DATA</span>}
          </div>
          <div className="table-scroll">
            <table className="baseline-table">
              <thead>
                <tr><th>APPROACH</th><th>RECALL@{dataset?.k ?? "K"}</th><th>P50 LATENCY</th><th>QPS</th><th>MEMORY</th></tr>
              </thead>
              <tbody>
                {baselineMethods.map((method) => {
                  const result = results.get(method.id);
                  return (
                    <tr key={method.id} className={method.id === "khnsw" ? "khnsw-row" : ""}>
                      <td><strong>{method.label}</strong>{method.id === "khnsw" && <small>Proposed</small>}</td>
                      <td>{formatBaselineMetric(result?.recallAtK, "recallAtK")}</td>
                      <td>{formatBaselineMetric(result?.p50LatencyMs, "p50LatencyMs")}</td>
                      <td>{formatBaselineMetric(result?.qps, "qps")}</td>
                      <td>{formatBaselineMetric(result?.memoryMb, "memoryMb")}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="baseline-footnote">
            <span>Recall is entered as a fraction from 0 to 1.</span>
            <span>Latency is p50 in ms; memory is MB.</span>
          </div>
          {!dataset && <p className="no-results-note">Import measured results to populate this comparison.</p>}
        </section>
        <section className="panel baseline-chart-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">VISUAL COMPARISON</span><h2>{baselineMetricLabels[metric]}</h2></div>
            <select aria-label="Chart metric" value={metric} onChange={(event) => setMetric(event.target.value as BaselineMetricKey)}>
              {(Object.entries(baselineMetricLabels) as [BaselineMetricKey, string][]).map(([key, label]) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>
          </div>
          <div className="chart-wrap baseline-chart">
            {chartData.length ? (
              <Suspense fallback={<div className="chart-loading">Loading chart…</div>}>
                <ChartPanel type="baseline" data={chartData} unit={chartUnit} />
              </Suspense>
            ) : (
              <div className="chart-empty">Import results containing {baselineMetricLabels[metric].toLowerCase()} to compare methods.</div>
            )}
          </div>
          <p className="baseline-provenance">
            {dataset ? "One uploaded run; compare only methods measured under the same setup." : "No benchmark values loaded."}
          </p>
        </section>
      </div>
    </>
  );
}

interface TraceWorkerResponse {
  id: number;
  events?: Event[];
  error?: string;
}

function LiveMetricsView({
  events,
  setEvents,
  traceName,
  setTraceName,
}: {
  events: Event[] | null;
  setEvents: (events: Event[]) => void;
  traceName: string;
  setTraceName: (name: string) => void;
}) {
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  const pendingTraceNameRef = useRef("Uploaded JSONL trace");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [cursor, setCursor] = useState(-1);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    const worker = new Worker(new URL("./worker/trace.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    worker.onmessage = (message: MessageEvent<TraceWorkerResponse>) => {
      if (message.data.id !== requestIdRef.current) return;
      setLoading(false);
      if (message.data.error) {
        setError(message.data.error);
        return;
      }
      if (!message.data.events) {
        setError("Trace worker returned no parsed events.");
        return;
      }
      setEvents(message.data.events);
      setTraceName(pendingTraceNameRef.current);
      setCursor(message.data.events.length - 1);
      setError("");
    };
    worker.onerror = (event) => {
      setLoading(false);
      setError(`Trace worker failed: ${event.message || "unknown worker error"}`);
    };
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [setEvents, setTraceName]);

  useEffect(() => {
    if (!playing || !events) return;
    const timer = window.setTimeout(() => {
      if (cursor >= events.length - 1) {
        setPlaying(false);
        return;
      }
      setCursor(cursor + 1);
    }, 350);
    return () => window.clearTimeout(timer);
  }, [cursor, events, playing]);

  const visibleEvents = useMemo(
    () => events?.slice(0, cursor + 1) ?? [],
    [cursor, events],
  );
  const metrics = useMemo(() => deriveLiveMetrics(visibleEvents), [visibleEvents]);
  const additions = visibleEvents.filter((event) => event.type === "edge_add").length;
  const removals = visibleEvents.filter((event) => event.type === "edge_remove").length;

  const handleTraceUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) return;
    const worker = workerRef.current;
    if (!worker) {
      setError("Trace worker is not ready. Try importing the file again.");
      return;
    }
    setError("");
    setLoading(true);
    setPlaying(false);
    try {
      const text = await file.text();
      requestIdRef.current += 1;
      pendingTraceNameRef.current = file.name;
      worker.postMessage({ id: requestIdRef.current, text });
    } catch (caught) {
      setLoading(false);
      setError(caught instanceof Error ? `Could not read trace file: ${caught.message}` : "Could not read trace file.");
    } finally {
      input.value = "";
    }
  };

  const togglePlayback = () => {
    if (!events?.length) return;
    if (!playing && cursor >= events.length - 1) setCursor(0);
    setPlaying(!playing);
  };
  const recallLabel = metrics.recall === null ? "—" : `${(metrics.recall * 100).toFixed(2)}%`;
  const latencyLabel = metrics.meanLatencyMs === null ? "—" : `${metrics.meanLatencyMs.toFixed(3)} ms`;
  const connectivityLabel = metrics.connectivity === null ? "—" : `${(metrics.connectivity * 100).toFixed(1)}%`;
  const processingLabel = metrics.processingSeconds === null ? "—" : `${metrics.processingSeconds.toFixed(3)} s`;
  const qpsLabel = metrics.qps === null ? "—" : metrics.qps.toFixed(2);
  const recentEvents = [...visibleEvents].reverse().slice(0, 7);
  const traceRunId = events?.[0]?.run_id ?? metrics.runId;
  const syntheticTrace = events?.some((event) => event.payload.example === true) ?? false;

  return (
    <>
      <section className="panel trace-controls">
        <div className="trace-source-copy">
          <span className="eyebrow">EVENT TRACE</span>
          <strong>{events ? traceName : "No trace loaded"}</strong>
          <small>{events ? `${events.length} validated events · run ${traceRunId ?? "—"}` : "Import a .jsonl file matching the event schema."}</small>
        </div>
        <label className="upload-button">
          <input type="file" accept=".jsonl,.json,text/plain,application/json" aria-label="Upload event trace" onChange={handleTraceUpload} />
          <span>{loading ? "Validating…" : "↑  Load trace"}</span>
        </label>
        <div className="trace-playback">
          <button type="button" className="play-button" onClick={togglePlayback} disabled={!events} aria-label={playing ? "Pause trace" : "Play trace"}>
            {playing ? "Ⅱ" : "▶"}
          </button>
          <input
            type="range"
            aria-label="Trace event position"
            min="0"
            max={Math.max(0, (events?.length ?? 0) - 1)}
            value={Math.max(0, cursor)}
            disabled={!events?.length}
            onChange={(event) => { setPlaying(false); setCursor(Number(event.target.value)); }}
          />
          <span>{events ? `${Math.max(0, cursor + 1)} / ${events.length}` : "0 / 0"}</span>
        </div>
      </section>
      {error && <div className="import-error" role="alert">{error}</div>}
      {syntheticTrace && (
        <div className="synthetic-warning">
          This trace is marked as a schema example. Its values are synthetic and are not benchmark results.
        </div>
      )}
      <div className="metrics-grid live-metrics-grid">
        <MetricCard label={metrics.recallName ?? "Recall"} value={recallLabel} caption="Latest metric event" icon="◎" tone="violet" />
        <MetricCard label="Mean query latency" value={latencyLabel} caption={`${metrics.queryCount} search events`} icon="⌁" tone="blue" />
        <MetricCard label="Repair count" value={String(metrics.repairCount)} caption="Scheduler pop events" icon="↻" tone="mint" />
        <MetricCard label="Modified edges" value={String(metrics.modifiedEdges)} caption={`${additions} added · ${removals} removed`} icon="⌘" tone="rose" />
        <MetricCard label="Connectivity" value={connectivityLabel} caption="Latest connectivity metric" icon="◉" tone="mint" />
        <MetricCard label="Processing time" value={processingLabel} caption="run_end elapsed_seconds" icon="◷" tone="blue" />
      </div>
      <div className="live-details-grid">
        <section className="panel live-summary-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">ENGINE SUMMARY</span><h2>Current trace state</h2></div>
            <span className={events ? "state-pill healthy-pill" : "state-pill"}><i /> {events ? "Trace loaded" : "Waiting for trace"}</span>
          </div>
          <div className="live-facts">
            <div><span>Events processed</span><strong>{metrics.eventCount}</strong></div>
            <div><span>Query throughput</span><strong>{qpsLabel}{metrics.qps === null ? "" : " QPS"}</strong></div>
            <div><span>Trace timestamp</span><strong>{visibleEvents.at(-1)?.t.toFixed(3) ?? "—"} s</strong></div>
            <div><span>Run identifier</span><strong>{metrics.runId ?? "—"}</strong></div>
          </div>
          <div className="live-explainer">
            <span>i</span>
            <p>Metrics are derived from events seen at the current replay position. Missing fields are shown as —.</p>
          </div>
        </section>
        <section className="panel event-feed-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">TRACE ACTIVITY</span><h2>Recent events</h2></div>
            <span className="event-count">{recentEvents.length} shown</span>
          </div>
          {recentEvents.length ? (
            <div className="event-feed">
              {recentEvents.map((event) => (
                <div className="event-feed-row" key={`${event.run_id}:${event.seq}`}>
                  <span className={`event-type event-${event.type}`}>{event.type.replace("_", " ")}</span>
                  <span>{event.node === null ? "—" : `node ${event.node}`}</span>
                  <time>{event.t.toFixed(3)}s</time>
                </div>
              ))}
            </div>
          ) : (
            <p className="chart-empty">Load a valid trace to inspect event activity.</p>
          )}
        </section>
      </div>
    </>
  );
}

function PriorityView({
  budget,
  setBudget,
}: {
  budget: number;
  setBudget: (budget: number) => void;
}) {
  const schedule = useMemo(() => createRepairSchedule(budget), [budget]);
  const costUsed = schedule.filter((item) => item.fitsBudget).reduce((sum, item) => sum + item.repairCost, 0);
  return (
    <>
      <section className="panel budget-panel">
        <div className="budget-intro">
          <div className="budget-icon">◷</div>
          <div><span className="eyebrow">REPAIR BUDGET</span><h2>Distance computation limit</h2><p>Adjust the available work for this demo schedule.</p></div>
        </div>
        <div className="budget-slider-wrap">
          <div className="budget-values"><span>Available budget</span><strong>{budget}<small> units</small></strong></div>
          <input
            aria-label="Repair budget"
            type="range"
            min="0"
            max="120"
            step="5"
            value={budget}
            onChange={(event) => setBudget(Number(event.target.value))}
          />
          <div className="budget-scale"><span>0</span><span>120 units</span></div>
        </div>
        <div className="budget-remaining"><span>Allocated</span><strong>{costUsed}<small> / {budget}</small></strong><div className="budget-progress"><i style={{ width: `${budget ? Math.min(100, (costUsed / budget) * 100) : 0}%` }} /></div></div>
      </section>
      <div className="priority-grid">
        <section className="panel schedule-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">URGENCY RANKING</span><h2>Repair candidates</h2></div>
            <span className="demo-tag">DEMO ESTIMATES</span>
          </div>
          <div className="table-scroll">
            <table className="priority-table">
              <thead><tr><th>RANK</th><th>NODE</th><th>PRIORITY</th><th>RECALL LOSS</th><th>COST</th><th>SCHEDULE</th></tr></thead>
              <tbody>
                {schedule.map((node, index) => (
                  <tr key={node.id}>
                    <td><span className="rank-number">{numberLabel(index + 1)}</span></td>
                    <td><strong className="table-node">{node.id}</strong></td>
                    <td><span className="priority-score"><i style={{ width: `${node.priority}%` }} />{node.priority}</span></td>
                    <td>{node.recallLoss.toFixed(2)}<small className="percent-sign">%</small></td>
                    <td>{node.repairCost}<small className="unit-sign"> units</small></td>
                    <td><span className={node.fitsBudget ? "fit-pill" : "wait-pill"}><i />{node.fitsBudget ? "Fits budget" : "Deferred"}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="table-footnote"><span><i className="legend-dot selected" /> Scheduled</span><span><i className="legend-dot muted" /> Deferred until budget allows</span></div>
          <p className="small-disclaimer">Recall-loss figures are illustrative estimates, not measured recall.</p>
        </section>
        <section className="panel urgency-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">QUEUE OVERVIEW</span><h2>Priority scores</h2></div>
          </div>
          <div className="chart-wrap priority-chart">
            <Suspense fallback={<div className="chart-loading">Loading chart…</div>}>
              <ChartPanel type="priority" data={schedule} />
            </Suspense>
          </div>
          <div className="queue-summary">
            <div><span>Repairs in budget</span><strong>{schedule.filter((item) => item.fitsBudget).length}<small> / {schedule.length}</small></strong></div>
            <div><span>Budget remaining</span><strong>{budget - costUsed}<small> units</small></strong></div>
          </div>
          <div className="scheduler-note"><span>↗</span><p>Repairs are ranked by priority; each task consumes its listed distance-computation cost.</p></div>
        </section>
      </div>
    </>
  );
}

export function App() {
  const [view, setView] = useState<View>("graph");
  const [layer, setLayer] = useState(0);
  const [selectedNode, setSelectedNode] = useState("N-08");
  const [movementMode, setMovementMode] = useState<MovementMode>("linear");
  const [time, setTime] = useState(3.2);
  const [playing, setPlaying] = useState(false);
  const [budget, setBudget] = useState(55);
  const [baselineDataset, setBaselineDataset] = useState<BaselineDataset | null>(null);
  const [traceEvents, setTraceEvents] = useState<Event[] | null>(null);
  const [traceName, setTraceName] = useState("");

  const selectLayer = (nextLayer: number) => {
    setLayer(nextLayer);
    const selected = playgroundNodes.find((node) => node.id === selectedNode);
    if (selected && selected.layer < nextLayer) {
      const visibleNode = playgroundNodes.find((node) => node.layer >= nextLayer);
      if (visibleNode) setSelectedNode(visibleNode.id);
    }
  };
  const selectNode = (nodeId: string) => {
    setSelectedNode(nodeId);
    const selected = playgroundNodes.find((node) => node.id === nodeId);
    if (selected && selected.layer < layer) setLayer(selected.layer);
  };

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setTime((current) => {
        if (current >= 12) {
          setPlaying(false);
          return 0;
        }
        return Math.min(12, Number((current + 0.1).toFixed(1)));
      });
    }, 140);
    return () => window.clearInterval(timer);
  }, [playing]);

  const { title, eyebrow, description } = viewTitles[view];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="K-HNSW home" onClick={(event) => { event.preventDefault(); setView("graph"); }}>
          <BrandMark />
          <span>K-HNSW<small>GRAPH STUDIO</small></span>
        </a>
        <div className="sidebar-section-label">WORKSPACE</div>
        <nav className="primary-nav" aria-label="Main navigation">
          {([
            ["graph", "Graph playground"],
            ["movement", "Movement simulator"],
            ["violations", "Violation view"],
            ["priority", "Priority view"],
            ["rewiring", "Rewiring view"],
            ["baseline", "Baseline vs K-HNSW"],
            ["metrics", "Live metrics"],
          ] as [View, string][]).map(([id, label]) => (
            <button
              type="button"
              key={id}
              className={view === id ? "nav-item active" : "nav-item"}
              onClick={() => setView(id)}
              aria-current={view === id ? "page" : undefined}
            >
              <NavigationIcon view={id} />
              <span>{label}</span>
              {id === "violations" && <em>04</em>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-model">
            <span className="model-indicator"><i /></span>
            <div><strong>Demo environment</strong><small>Local illustrative data</small></div>
            <span className="more-dots">···</span>
          </div>
          <div className="user-profile">
            <span className="avatar">KH</span>
            <div><strong>Graph explorer</strong><small>Research workspace</small></div>
            <span className="more-dots">⌄</span>
          </div>
        </div>
      </aside>
      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs"><span>WORKSPACE</span><b>/</b><strong>{eyebrow}</strong></div>
          <div className="topbar-actions"><span className="sample-run"><i /> DEMO TRACE</span><button type="button" className="help-button" aria-label="Demo data information">i</button></div>
        </header>
        <div className="page-content">
          <div className="page-heading">
            <div>
              <span className="eyebrow">{eyebrow}</span>
              <h1>{title}</h1>
              <p>{description}</p>
            </div>
            <button type="button" className="trace-button" onClick={() => { setTime(0); setView("movement"); setPlaying(false); }}>
              <span>↻</span> Replay demo
            </button>
          </div>
          {view === "graph" && (
            <GraphView
              layer={layer}
              setLayer={selectLayer}
              selectedNode={selectedNode}
              setSelectedNode={selectNode}
              movementMode={movementMode}
              time={time}
            />
          )}
          {view === "movement" && (
            <MovementView
              layer={layer}
              setLayer={selectLayer}
              selectedNode={selectedNode}
              setSelectedNode={selectNode}
              movementMode={movementMode}
              setMovementMode={setMovementMode}
              time={time}
              setTime={setTime}
              playing={playing}
              setPlaying={setPlaying}
            />
          )}
          {view === "violations" && (
            <ViolationView
              layer={layer}
              setLayer={selectLayer}
              selectedNode={selectedNode}
              setSelectedNode={selectNode}
              movementMode={movementMode}
              time={time}
            />
          )}
          {view === "priority" && <PriorityView budget={budget} setBudget={setBudget} />}
          {view === "rewiring" && (
            <RewiringView
              layer={layer}
              setLayer={selectLayer}
              selectedNode={selectedNode}
              setSelectedNode={selectNode}
              movementMode={movementMode}
              time={time}
            />
          )}
          {view === "baseline" && (
            <BaselineView dataset={baselineDataset} setDataset={setBaselineDataset} />
          )}
          {view === "metrics" && (
            <LiveMetricsView
              events={traceEvents}
              setEvents={setTraceEvents}
              traceName={traceName}
              setTraceName={setTraceName}
            />
          )}
        </div>
        <footer className="app-footer">
          <span>KHNSW <i>·</i> GRAPH STUDIO</span>
          <span>Sample visualizer — not connected to a live index</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
