import time
import json
from typing import Dict, Any

from vrp_solver import BaseSolver

try:
    from cuopt_sh_client import CuOptServiceSelfHostClient
except Exception:
    CuOptServiceSelfHostClient = None


class SolverCuOpt(BaseSolver):
    """Submit a CSR-formatted MIP to a cuOpt self-host service and parse the response.

    The solver accepts an `instance` dict that either contains a ready-to-send
    CSR MIP payload (keys: 'csr_constraint_matrix', 'constraint_bounds',
    'objective_data', 'variable_bounds', 'variable_names', 'variable_types') or a
    VRP-style instance (with 'nodes', 'customers', 'num_vehicles', 'distance_matrix', etc.)
    that this class will convert to a minimal CSR form similar to
    `solve/dispatch_mip_cuopt.py`.
    """

    def __init__(self, ip: str = "0.0.0.0", port: int = 5001, polling_timeout: int = 25):
        if CuOptServiceSelfHostClient is None:
            raise ImportError("cuopt_sh_client is not installed")
        # create client but delay until solve so callers can instantiate without network
        self.ip = ip
        self.port = port
        self.polling_timeout = polling_timeout

    def _build_payload_from_instance(self, instance: Dict) -> Dict:
        # If instance already contains CSR payload fields, assume user prepared it
        if all(k in instance for k in ("csr_constraint_matrix", "constraint_bounds", "objective_data", "variable_bounds", "variable_names", "variable_types")):
            payload = {
                "csr_constraint_matrix": instance["csr_constraint_matrix"],
                "constraint_bounds": instance["constraint_bounds"],
                "objective_data": instance["objective_data"],
                "variable_bounds": instance["variable_bounds"],
                "maximize": instance.get("maximize", False),
                "variable_names": instance["variable_names"],
                "variable_types": instance["variable_types"],
                "solver_config": instance.get("solver_config", {"time_limit": int(instance.get("time_limit", 30))})
            }
            return payload

        # Otherwise attempt a best-effort conversion for VRP-like instances
        # This mirrors a minimal subset of the construction used in dispatch_mip_cuopt.py
        nodes = instance.get("nodes")
        num_vehicles = instance.get("num_vehicles")
        depot = instance.get("depot", 0)
        customers = instance.get("customers")
        distance_matrix = instance.get("distance_matrix")
        demands = instance.get("demands")
        vehicle_capacities = instance.get("vehicle_capacities")

        if not (nodes and num_vehicles and customers and distance_matrix and demands and vehicle_capacities):
            raise ValueError("Instance must include CSR fields or at least nodes/num_vehicles/customers/distance_matrix/demands/vehicle_capacities")

        # Build variable names (x, y, a, u, delta_plus, delta_minus, z) similar to dispatch
        variable_names = []
        variable_types = []
        var_lb = []
        var_ub = []

        x_indices = []
        for k in range(num_vehicles):
            for i in nodes:
                for j in nodes:
                    if i != j:
                        variable_names.append(f"x_{i}_{j}_{k}")
                        variable_types.append("I")
                        var_lb.append(0)
                        var_ub.append(1)
                        x_indices.append((i, j, k))

        y_indices = []
        for k in range(num_vehicles):
            variable_names.append(f"y_{k}")
            variable_types.append("I")
            var_lb.append(0)
            var_ub.append(1)
            y_indices.append(k)

        a_indices = []
        for i in nodes:
            variable_names.append(f"a_{i}")
            variable_types.append("C")
            var_lb.append(0)
            var_ub.append(1e6)
            a_indices.append(i)

        u_indices = []
        for i in nodes:
            for k in range(num_vehicles):
                variable_names.append(f"u_{i}_{k}")
                variable_types.append("C")
                var_lb.append(0)
                var_ub.append(vehicle_capacities[k])
                u_indices.append((i, k))

        delta_plus_indices = []
        delta_minus_indices = []
        for i in customers:
            variable_names.append(f"delta_plus_{i}")
            variable_types.append("C")
            var_lb.append(0)
            var_ub.append(1e6)
            delta_plus_indices.append(i)
            variable_names.append(f"delta_minus_{i}")
            variable_types.append("C")
            var_lb.append(0)
            var_ub.append(1e6)
            delta_minus_indices.append(i)

        z_indices = []
        for i in customers:
            variable_names.append(f"z_{i}")
            variable_types.append("I")
            var_lb.append(0)
            var_ub.append(1)
            z_indices.append(i)

        # Build CSR (minimal subset): start constraints, return constraints, visit==z and flow conservation
        csr_offsets = [0]
        csr_indices = []
        csr_values = []
        upper_bounds = []
        lower_bounds = []

        # each vehicle leaves depot -> equals y_k
        for k in range(num_vehicles):
            row_idx = []
            row_val = []
            for j in nodes:
                if j != depot:
                    idx = variable_names.index(f"x_{depot}_{j}_{k}")
                    row_idx.append(idx)
                    row_val.append(1.0)
            idx_y = variable_names.index(f"y_{k}")
            row_idx.append(idx_y)
            row_val.append(-1.0)
            csr_indices.extend(row_idx)
            csr_values.extend(row_val)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(0.0)
            lower_bounds.append(0.0)

        # each vehicle returns to depot
        for k in range(num_vehicles):
            row_idx = []
            row_val = []
            for i in nodes:
                if i != depot:
                    idx = variable_names.index(f"x_{i}_{depot}_{k}")
                    row_idx.append(idx)
                    row_val.append(1.0)
            idx_y = variable_names.index(f"y_{k}")
            row_idx.append(idx_y)
            row_val.append(-1.0)
            csr_indices.extend(row_idx)
            csr_values.extend(row_val)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(0.0)
            lower_bounds.append(0.0)

        # each customer visit == z_i
        for i in customers:
            row_idx = []
            row_val = []
            for k in range(num_vehicles):
                for j in nodes:
                    if j != i:
                        idx = variable_names.index(f"x_{j}_{i}_{k}")
                        row_idx.append(idx)
                        row_val.append(1.0)
            idx_z = variable_names.index(f"z_{i}")
            row_idx.append(idx_z)
            row_val.append(-1.0)
            csr_indices.extend(row_idx)
            csr_values.extend(row_val)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(0.0)
            lower_bounds.append(0.0)

        # flow conservation for nodes
        for k in range(num_vehicles):
            for i in nodes:
                row_idx = []
                row_val = []
                for j in nodes:
                    if j != i:
                        idx_in = variable_names.index(f"x_{j}_{i}_{k}")
                        idx_out = variable_names.index(f"x_{i}_{j}_{k}")
                        row_idx.append(idx_in)
                        row_val.append(1.0)
                        row_idx.append(idx_out)
                        row_val.append(-1.0)
                csr_indices.extend(row_idx)
                csr_values.extend(row_val)
                csr_offsets.append(len(csr_indices))
                upper_bounds.append(0.0)
                lower_bounds.append(0.0)

        # objective: build coefficients for variable_names
        coeffs = [0.0] * len(variable_names)
        w_fixed = instance.get("w_fixed", 1.0)
        w_distance = instance.get("w_distance", 1.0)
        w_early = instance.get("w_early", 1.0)
        w_late = instance.get("w_late", 1.0)
        w_unserved = instance.get("w_unserved", 1.0)
        vehicle_cost_per_km = instance.get("vehicle_cost_per_km", [1.0] * num_vehicles)
        beta = instance.get("beta", 10)
        gamma = instance.get("gamma", 20)
        lambda_demand = instance.get("lambda_demand", [0]*len(nodes))
        vehicle_fixed_costs = instance.get("vehicle_fixed_costs", [0]*num_vehicles)

        for k in range(num_vehicles):
            idx = variable_names.index(f"y_{k}")
            coeffs[idx] += w_fixed * vehicle_fixed_costs[k]
        for (i, j, k) in x_indices:
            idx = variable_names.index(f"x_{i}_{j}_{k}")
            coeffs[idx] += w_distance * vehicle_cost_per_km[k] * float(distance_matrix[i][j])
        for i in customers:
            idx_plus = variable_names.index(f"delta_plus_{i}")
            idx_minus = variable_names.index(f"delta_minus_{i}")
            coeffs[idx_plus] += w_early * beta
            coeffs[idx_minus] += w_late * gamma
            idx_z = variable_names.index(f"z_{i}")
            coeffs[idx_z] -= w_unserved * lambda_demand[i]

        payload = {
            "csr_constraint_matrix": {"offsets": csr_offsets, "indices": csr_indices, "values": csr_values},
            "constraint_bounds": {"upper_bounds": upper_bounds, "lower_bounds": lower_bounds},
            "objective_data": {"coefficients": coeffs, "scalability_factor": 1.0, "offset": 0.0},
            "variable_bounds": {"upper_bounds": var_ub, "lower_bounds": var_lb},
            "maximize": False,
            "variable_names": variable_names,
            "variable_types": variable_types,
            "solver_config": {"time_limit": int(instance.get("time_limit", 30))}
        }
        return payload

    def _repoll(self, client, solution, repoll_tries=500, sleep_s=1):
        if isinstance(solution, dict) and "reqId" in solution and "response" not in solution:
            req_id = solution["reqId"]
            for _ in range(repoll_tries):
                solution = client.repoll(req_id, response_type="dict")
                if isinstance(solution, dict) and "response" in solution:
                    break
                time.sleep(sleep_s)
        return solution

    def solve(self, instance: Dict, time_limit: float = 30.0) -> Dict[str, Any]:
        if CuOptServiceSelfHostClient is None:
            return {"status": "cuopt_client_unavailable", "objective": None, "elapsed": 0}

        client = CuOptServiceSelfHostClient(ip=self.ip, port=self.port, polling_timeout=self.polling_timeout, timeout_exception=False)

        payload = self._build_payload_from_instance({**instance, "time_limit": time_limit})

        # prefer get_MIP_solve if available, fallback to get_LP_solve
        start = time.time()
        if hasattr(client, "get_MIP_solve"):
            sol = client.get_MIP_solve(payload, response_type="dict")
        else:
            sol = client.get_LP_solve(payload, response_type="dict")
        sol = self._repoll(client, sol)
        elapsed = time.time() - start

        result = {
            "objective": None,
            "elapsed": elapsed,
            "used_vehicles": None,
            "gap": None,
            "best_bound": None,
            "nodes": None,
            "memory": None,
            "status": "NO_RESPONSE",
            "vars": {}
        }

        if isinstance(sol, dict) and "response" in sol:
            resp = sol["response"]
            solver_resp = resp.get("solver_response", {})
            solution = solver_resp.get("solution", {})
            result["vars"] = solution.get("vars", {})
            result["objective"] = solution.get("primal_objective", solution.get("objective", None))
            # try to get used vehicles by counting y_k variables if present
            vars_map = result["vars"] or {}
            used = 0
            for k in range(instance.get("num_vehicles", 0)):
                val = vars_map.get(f"y_{k}")
                try:
                    if val is not None and float(val) > 0.5:
                        used += 1
                except Exception:
                    pass
            result["used_vehicles"] = used
            result["status"] = resp.get("status", "OK")
            # best_bound / gap may be in solver_response
            result["best_bound"] = solver_resp.get("best_bound") or solver_resp.get("best_lower_bound")
            result["gap"] = solver_resp.get("gap")
        else:
            result["status"] = "NO_RESPONSE"

        return result
