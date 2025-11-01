import time
from typing import Dict, Any
from solver.vrp_solver import BaseSolver

try:
    from cuopt_sh_client import CuOptServiceSelfHostClient
except Exception:
    CuOptServiceSelfHostClient = None


class SolverCuOptVRP(BaseSolver):
    def __init__(self, ip: str = "localhost", port: int = 5001):
        if CuOptServiceSelfHostClient is None:
            raise ImportError("cuopt_sh_client is not available")
        self.client = CuOptServiceSelfHostClient(ip=ip, port=port, polling_timeout=25, timeout_exception=False)

    def repoll(self, solution, repoll_tries=500):
        if "reqId" in solution and "response" not in solution:
            req_id = solution["reqId"]
            for _ in range(repoll_tries):
                solution = self.client.repoll(req_id, response_type="dict")
                if "reqId" in solution and "response" in solution:
                    break
        return solution

    def solve(self, instance: Dict, time_limit: float = 30.0) -> Dict[str, Any]:
        if CuOptServiceSelfHostClient is None:
            return {"status": "cuopt_client_unavailable", "objective": None, "elapsed": 0}

        # Build routing request payload similar to dispatch_vrp_cuopt
        num_vehicles = instance["num_vehicles"]
        depot = instance.get("depot", 0)
        customers = instance["customers"]
        distance_matrix = instance["distance_matrix"]
        demands = instance["demands"]
        vehicle_capacities = instance["vehicle_capacities"]
        vehicle_fixed_costs = instance.get("vehicle_fixed_costs", [0]*num_vehicles)
        vehicle_speeds = instance.get("vehicle_speeds", [1]*num_vehicles)
        vehicle_cost_per_km = instance.get("vehicle_cost_per_km", [1]*num_vehicles)
        time_windows = instance.get("time_windows", [])
        service_times = instance.get("service_times", [])
        beta = instance.get("beta", 10)
        gamma = instance.get("gamma", 20)

        data = {
            "cost_matrix_data": {"data": {"0": distance_matrix}},
            "task_data": {
                "task_locations": customers,
                "demand": [demands[i] for i in customers],
                "task_time_windows": [list(time_windows[i]) for i in customers] if time_windows else [],
                "service_times": [service_times[i] for i in customers] if service_times else []
            },
            "fleet_data": {
                "vehicle_locations": [[depot] for _ in range(num_vehicles)],
                "capacities": vehicle_capacities,
                "fixed_costs": vehicle_fixed_costs,
                "speeds": vehicle_speeds,
                "cost_per_distance": vehicle_cost_per_km,
                "time_windows": [list(time_windows[depot]) for _ in range(num_vehicles)] if time_windows else []
            },
            "penalties": {"early": beta, "late": gamma}
        }

        start = time.time()
        solution = self.client.get_optimized_routes(data)
        solution = self.repoll(solution)
        elapsed = time.time() - start

        objective = None
        used_vehicles = 0
        if isinstance(solution, dict) and "response" in solution:
            resp = solution["response"]
            objective = resp.get("objective_value")
            routes = resp.get("routes", [])
            used_vehicles = sum(1 for r in routes if len(r) > 2)
            status = "OK"
        else:
            status = "NO_RESPONSE"

        return {
            "objective": objective,
            "elapsed": elapsed,
            "used_vehicles": used_vehicles,
            "gap": None,
            "best_bound": None,
            "nodes": None,
            "memory": None,
            "status": status,
            "vars": solution.get("response") if isinstance(solution, dict) else {}
        }
