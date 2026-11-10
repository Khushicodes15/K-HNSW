# Movement Models (`src/khnsw/core/movement`)

## Purpose
Simulates continuous or discrete spatial movement of embedded vector points over time ($t \ge 0.0$), generating deterministic trajectory vectors for kinetic evaluations.

## Owner Role
**Member 3 (Update/Repair Engine - Repo Owner)**

## What Goes Here
- Implementations of the `MovementModel` protocol:
  - Random walk / Brownian diffusion with controlled step sizes.
  - Clustered trajectory drift (coherent movement of vector semantic clusters).
  - Rotational and oscillatory kinematics.
- Vector velocity and coordinate delta computation utilities.
- Trajectory validation tools verifying shapes and bounding constraints.

