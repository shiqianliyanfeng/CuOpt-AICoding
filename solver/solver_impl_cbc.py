import time
import numpy as np
from ortools.linear_solver import pywraplp
from typing import Dict, Any
from solver.vrp_solver import BaseSolver

class SolverCBC(BaseSolver):
    def __init__(self):
        pass

    def solve(self, instance: Dict, time_limit: float = 30.0) -> Dict[str, Any]:
        num_vehicles = instance["num_vehicles"]
        nodes = instance["nodes"]
        customers = instance["customers"]
        depot = instance.get("depot", 0)
        demands = instance["demands"]
        vehicle_capacities = instance["vehicle_capacities"]
        distance_matrix = np.array(instance["distance_matrix"])
        vehicle_fixed_costs = instance.get("vehicle_fixed_costs", [0]*num_vehicles)
        vehicle_cost_per_km = instance.get("vehicle_cost_per_km", [1.0]*num_vehicles)
        time_windows = instance.get("time_windows", [])
        service_times = instance.get("service_times", [0]*len(nodes))
        vehicle_speeds = instance.get("vehicle_speeds", [1]*num_vehicles)
        beta = instance.get("beta", 10)
        gamma = instance.get("gamma", 20)
        lambda_demand = instance.get("lambda_demand", [0]*len(nodes))
        w_fixed = instance.get("w_fixed", 1.0)
        w_distance = instance.get("w_distance", 1.0)
        w_early = instance.get("w_early", 1.0)
        w_late = instance.get("w_late", 1.0)
        w_unserved = instance.get("w_unserved", 1.0)

        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return {"status": "solver_unavailable", "objective": None, "elapsed": 0}
        # time_limit in seconds -> milliseconds for SetTimeLimit
        solver.SetTimeLimit(int(max(1, time_limit) * 1000))

        # variables
        x = {}
        for k in range(num_vehicles):
            for i in nodes:
                for j in nodes:
                    if i != j:
                        x[i, j, k] = solver.BoolVar(f"x_{i}_{j}_{k}")
        y = [solver.BoolVar(f"y_{k}") for k in range(num_vehicles)]
        a = {i: solver.NumVar(0, solver.infinity(), f"a_{i}") for i in nodes}
        u = {}
        for i in nodes:
            for k in range(num_vehicles):
                u[i, k] = solver.NumVar(0, vehicle_capacities[k], f"u_{i}_{k}")
        delta_plus = {i: solver.NumVar(0, solver.infinity(), f"delta_plus_{i}") for i in customers}
        delta_minus = {i: solver.NumVar(0, solver.infinity(), f"delta_minus_{i}") for i in customers}
        z = {i: solver.BoolVar(f"z_{i}") for i in customers}

        # constraints (essential subset)
        # vehicle start/return
        for k in range(num_vehicles):
            solver.Add(sum(x[depot, j, k] for j in nodes if j != depot) == y[k])
            solver.Add(sum(x[j, depot, k] for j in nodes if j != depot) == y[k])
        # customer visit == z
        for i in customers:
            solver.Add(sum(x[j, i, k] for k in range(num_vehicles) for j in nodes if j != i) == z[i])
        # flow conservation
        for k in range(num_vehicles):
            for i in nodes:
                solver.Add(sum(x[j, i, k] for j in nodes if j != i) == sum(x[i, j, k] for j in nodes if j != i))
        # capacity / simple form
        for k in range(num_vehicles):
            for j in customers:
                solver.Add(u[j, k] >= demands[j] * z[j])
                solver.Add(u[j, k] <= vehicle_capacities[k])
            solver.Add(u[depot, k] == 0)
        # time windows (big-M)
        M = 1e6
        for k in range(num_vehicles):
            for i in nodes:
                for j in customers:
                    if i != j:
                        travel_time = distance_matrix[i][j] / max(1e-6, vehicle_speeds[k])
                        solver.Add(a[j] - a[i] >= service_times[i] + travel_time - M * (1 - x[i, j, k]))
        if depot in a:
            solver.Add(a[depot] == 0)

        # objective
        obj_terms = []
        obj_terms += [w_fixed * vehicle_fixed_costs[k] * y[k] for k in range(num_vehicles)]
        obj_terms += [w_distance * vehicle_cost_per_km[k] * distance_matrix[i][j] * x[i, j, k]
                      for k in range(num_vehicles) for i in nodes for j in nodes if i != j]
        obj_terms += [w_early * beta * delta_plus[i] for i in customers]
        obj_terms += [w_late * gamma * delta_minus[i] for i in customers]
        obj_terms += [- w_unserved * lambda_demand[i] * z[i] for i in customers]
        solver.Minimize(solver.Sum(obj_terms))

        # solve
        start = time.time()
        status = solver.Solve()
        elapsed = time.time() - start

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            objective = solver.Objective().Value()
            used_vehicles = int(sum(1 for k in range(num_vehicles) if y[k].solution_value() > 0.5))
            status_str = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"
        else:
            objective = None
            used_vehicles = 0
            status_str = "INFEASIBLE_OR_ABORT"

        return {
            "objective": objective,
            "elapsed": elapsed,
            "used_vehicles": used_vehicles,
            "gap": None,
            "best_bound": None,
            "nodes": None,
            "memory": None,
            "status": status_str,
            "vars": {}  # detailed vars omitted to save memory; can be added if needed
        }