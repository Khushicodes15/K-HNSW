# Research & Engineering Task Roadmaps

This document outlines the project tasks for each team member. All tasks are tracked as checklists without calendar dates.

## Member 1: Research & Algorithm Specification
- [ ] Construct literature matrix covering dynamic HNSW variants and kinetic data structures
- [ ] Formulate formal mathematical problem definition for kinetic vector topology repair
- [ ] Derive geometric certificate equations and validity bounds
- [ ] Author priority scheduler specification and theoretical convergence proofs
- [ ] Perform asymptotic complexity analysis for certificate verification and local rewiring
- [ ] Conduct target venue research and formatting requirements from official conference sites
- [ ] Draft empirical go/no-go research memorandum based on preliminary baseline data

## Member 2: Graph Engine & Baselines
- [ ] Implement pure NumPy Hierarchical Navigable Small World (`GraphIndex`)
- [ ] Build reverse-edge index and incoming neighbor tracking (`in_neighbors`)
- [ ] Develop graph audit tools (layer degree bounds, entry point validity, connectivity checks)
- [ ] Implement hnswlib parity test suite verifying structural and recall alignment
- [ ] Implement delete-and-reinsert baseline adapter
- [ ] Implement complete periodic rebuild baseline adapter

## Member 3: Update & Repair Engine (Repo Owner)
- [ ] Implement synthetic kinetic movement models (random walk, clustered semantic drift)
- [ ] Implement kinetic certificate evaluation and invalidation detection
- [ ] Build versioned repair event heap with stale event filtering (`stale_skip`)
- [ ] Implement local 1-hop and 2-hop neighborhood repair routines
- [ ] Implement hierarchy-aware repair prioritizing upper routing layers
- [ ] Build budget-constrained priority scheduler
- [ ] Introduce configurable ablation switches (e.g. disabling upper-layer priority, disabling certificates)
- [ ] Integrate comprehensive event logging (`EventSink`) across all state transitions

## Member 4: Benchmarks & Parity Evaluation
- [ ] Design and run initial go/no-go experiment treating hnswlib as a black box
- [ ] Build brute-force exact ground truth generator for dynamic vector queries
- [ ] Implement benchmark metrics evaluation suite (Recall@k, latency percentiles, QPS, modified edge counters)
- [ ] Construct standardized benchmark execution harness across synthetic and real datasets
- [ ] Run ablation study matrix and measure repair trade-offs
- [ ] Compute summary statistics, confidence intervals, and variance metrics
- [ ] Implement simulation replay trace JSON export for frontend visualization

## Member 5: Frontend & System Integration
- [ ] Implement interactive dashboard views (graph topology view, latency/recall curves)
- [ ] Implement Pyodide web worker loader for client-side Python execution
- [ ] Build interactive replay player with step, play/pause, and event scrubbing
- [ ] Configure and test GitHub Pages automated deployment workflow
- [ ] Set up Playwright end-to-end browser smoke tests
- [ ] Maintain and verify Docker backend and Docker Compose multi-service configurations

