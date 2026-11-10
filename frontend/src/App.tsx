import "./App.css";

export function App() {
  return (
    <main className="container">
      <header>
        <h1>K-HNSW</h1>
        <p className="subtitle">
          Kinetic Hierarchical Navigable Small World: Research on repairing HNSW graph topology
          under vector updates
        </p>
      </header>

      <section className="status-card">
        <span className="badge">Status: Scaffold</span>
        <p>
          Repository scaffold initialized. Core interfaces, schemas, contracts, and tooling are in
          place. Algorithms and visualizations are pending implementation by team members.
        </p>
      </section>

      <section className="info-grid">
        <div className="card">
          <h2>Pure Core</h2>
          <p>
            Pure Python stdlib + NumPy algorithms in <code>src/khnsw/core/</code>, designed to run in
            the browser via Pyodide.
          </p>
        </div>
        <div className="card">
          <h2>Schemas & Contracts</h2>
          <p>
            Strict typing and event streams validated across Python (jsonschema) and TypeScript (Zod).
          </p>
        </div>
        <div className="card">
          <h2>Replay & Dashboard</h2>
          <p>
            Interactive timeline and topology inspection tools will be implemented in subsequent phases.
          </p>
        </div>
      </section>
    </main>
  );
}

export default App;
