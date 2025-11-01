# Vehicle Routing Problem (VRP) — Model (English)

This document describes the mixed-integer programming (MIP) model used by this repository.

## Notation
- V: set of nodes; 0 denotes the depot.
- C: set of customers (V \ {0}).
- K: set of vehicles.

Variables
- x_{i,j,k} ∈ {0,1}: 1 if vehicle k travels from node i to node j.
- y_k ∈ {0,1}: 1 if vehicle k is used.
- a_i ≥ 0: arrival time at node i.
- u_{i,k} ≥ 0: load of vehicle k after visiting node i.
- δ^+_i, δ^-_i ≥ 0: early / late slack for time windows.
- z_i ∈ {0,1}: 1 if customer i is served.

Parameters
- d_{i,j}: distance between node i and j.
- c^k_{i,j}: per-distance cost for vehicle k on arc (i,j).
- f_k: fixed cost to deploy vehicle k.
- Q_k: capacity of vehicle k.
- q_i: demand of customer i.
- s_i: service time at node i.
- [e_i, l_i]: time window for node i.
- β, γ: unit penalties for early/late arrival.
- λ_i: reward/penalty weight for not serving customer i.
- Weights: w_fixed, w_distance, w_early, w_late, w_unserved.

Objective
Minimize:

\[
w_{fixed} \sum_{k\in K} f_k y_k + w_{distance} \sum_{k\in K}\sum_{i\in V}\sum_{j\in V, j\ne i} c^k_{i,j} d_{i,j} x_{i,j,k} + w_{early} \beta \sum_{i\in C} \delta^+_i + w_{late} \gamma \sum_{i\in C} \delta^-_i - w_{unserved} \sum_{i\in C} \lambda_i z_i
\]

Constraints
1. Vehicle start/return at depot:
   - For each k: \sum_{j} x_{0,j,k} = y_k and \sum_{j} x_{j,0,k} = y_k.
2. Customer visit equals z_i:
   - For each i\in C: \sum_{k\in K}\sum_{j\in V\setminus\{i\}} x_{j,i,k} = z_i.
3. Flow conservation:
   - For each k and i: \sum_{j\ne i} x_{j,i,k} = \sum_{j\ne i} x_{i,j,k}.
4. Capacity propagation (example Big-M formulation):
   - u_{j,k} ≥ u_{i,k} + q_j z_j - M(1 - x_{i,j,k})
   - u_{0,k} = 0, and 0 ≤ u_{i,k} ≤ Q_k.
5. Time windows (Big-M soft enforcement):
   - a_j ≥ a_i + s_i + travel_time_{i,j} - M(1 - x_{i,j,k})
   - a_i - δ^+_i ≥ e_i; a_i + δ^-_i ≤ l_i.
6. Domains: x,y,z binary; a,u,δ continuous and nonnegative.

Notes
- Select Big-M values carefully; too large M leads to numerical instability.
- The model allows customers to be unserved (z_i = 0) and penalizes this via λ_i and w_unserved.
- The model can be exported to a CSR representation (variable names, types, bounds, constraint offsets/indices/values) for solver services.

---

Please refer to the code in `solver/` for example CSR assembly and API payloads.
