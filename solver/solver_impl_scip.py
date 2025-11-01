import time
import numpy as np
from typing import Dict, Any
from solver.vrp_solver import BaseSolver

try:
    import pyscipopt
except Exception:
    pyscipopt = None


class SolverSCIP(BaseSolver):
    def __init__(self):
        if pyscipopt is None:
            raise ImportError("pyscipopt is not available")

    def solve(self, instance: Dict, time_limit: float = 30.0) -> Dict[str, Any]:
        if pyscipopt is None:
            return {"status": "pyscipopt_unavailable", "objective": None, "elapsed": 0}

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

        model = pyscipopt.Model()
        model.setParam('display/verblevel', 0)
        model.setParam('limits/time', float(time_limit))

        # variables
        x = {}
        for k in range(num_vehicles):
            for i in nodes:
                for j in nodes:
                    if i != j:
                        x[i, j, k] = model.addVar(vtype='B', name=f"x_{i}_{j}_{k}")
        y = [model.addVar(vtype='B', name=f"y_{k}") for k in range(num_vehicles)]
        a = {i: model.addVar(lb=0.0, ub=None, vtype='C', name=f"a_{i}") for i in nodes}
        u = {}
        for i in nodes:
            for k in range(num_vehicles):
                u[i, k] = model.addVar(lb=0.0, ub=vehicle_capacities[k], vtype='C', name=f"u_{i}_{k}")
        delta_plus = {i: model.addVar(lb=0.0, ub=None, vtype='C', name=f"delta_plus_{i}") for i in customers}
        delta_minus = {i: model.addVar(lb=0.0, ub=None, vtype='C', name=f"delta_minus_{i}") for i in customers}
        z = {i: model.addVar(vtype='B', name=f"z_{i}") for i in customers}

        # constraints (same structure as other solvers)
        for k in range(num_vehicles):
            model.addCons(pyscipopt.quicksum(x[depot, j, k] for j in nodes if j != depot) == y[k])
            model.addCons(pyscipopt.quicksum(x[j, depot, k] for j in nodes if j != depot) == y[k])
        for i in customers:
            model.addCons(pyscipopt.quicksum(x[j, i, k] for k in range(num_vehicles) for j in nodes if j != i) == z[i])
        for k in range(num_vehicles):
            for i in nodes:
                model.addCons(pyscipopt.quicksum(x[j, i, k] for j in nodes if j != i) == pyscipopt.quicksum(x[i, j, k] for j in nodes if j != i))
        # capacity
        for k in range(num_vehicles):
            for j in customers:
                model.addCons(u[j, k] >= demands[j] * z[j])
                model.addCons(u[j, k] <= vehicle_capacities[k])
            model.addCons(u[depot, k] == 0)
        M = 1e6
        for k in range(num_vehicles):
            for i in nodes:
                for j in customers:
                    if i != j:
                        travel_time = float(distance_matrix[i][j]) / max(1e-6, vehicle_speeds[k])
                        model.addCons(a[j] - a[i] >= service_times[i] + travel_time - M * (1 - x[i, j, k]))
        model.addCons(a[depot] == 0)

        # objective
        obj = pyscipopt.quicksum(w_fixed * vehicle_fixed_costs[k] * y[k] for k in range(num_vehicles)) + \
              pyscipopt.quicksum(w_distance * vehicle_cost_per_km[k] * distance_matrix[i][j] * x[i, j, k]
                                 for k in range(num_vehicles) for i in nodes for j in nodes if i != j) + \
              pyscipopt.quicksum(w_early * beta * delta_plus[i] for i in customers) + \
              pyscipopt.quicksum(w_late * gamma * delta_minus[i] for i in customers) - \
              pyscipopt.quicksum(w_unserved * lambda_demand[i] * z[i] for i in customers)
        model.setObjective(obj, "minimize")

        # solve
        start = time.time()
        model.optimize()
        elapsed = time.time() - start

        status = model.getStatus()
        if status in ["optimal", "bestsollimit", "timelimit"]:
            try:
                objective = model.getObjVal()
            except Exception:
                objective = None
            used_vehicles = sum(1 for k in range(num_vehicles) if model.getVal(y[k]) > 0.5)
            nodes = model.getNNodes() if hasattr(model, 'getNNodes') else None
            best_bound = model.getLowerbound() if hasattr(model, 'getLowerbound') else None
            gap = model.getGap() if hasattr(model, 'getGap') else None
            memory = None
            status_str = status
        else:
            objective = None
            used_vehicles = 0
            nodes = None
            best_bound = None
            gap = None
            memory = None
            status_str = status

        return {
            "objective": objective,
            "elapsed": elapsed,
            "used_vehicles": used_vehicles,
            "gap": gap,
            "best_bound": best_bound,
            "nodes": nodes,
            "memory": memory,
            "status": status_str,
            "vars": {}
        }
