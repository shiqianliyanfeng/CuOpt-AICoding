from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import time
import numpy as np
import random
import matplotlib.pyplot as plt

class VRPMIPSolver:
    def __init__(self, ip="0.0.0.0", port=5001):
        self.cuopt_service_client = CuOptServiceSelfHostClient(
            ip=ip,
            port=port,
            polling_timeout=25,
            timeout_exception=False
        )
        self.model_data = self.gen_model_data()
        self.variable_names = []
        self.variable_types = []
        self.variable_bounds_lb = []
        self.variable_bounds_ub = []
        self.x_indices = []
        self.y_indices = []
        self.a_indices = []
        self.u_indices = []
        self.delta_plus_indices = []
        self.delta_minus_indices = []
        self.z_indices = []
        self._build_variables()
        self.problem_data = self._build_problem_data()

    def gen_model_data(self):
        num_customers = 4
        num_vehicles = 2
        depot = 0
        nodes = list(range(num_customers + 1))
        customers = nodes[1:]
        coords = np.random.rand(num_customers + 1, 2) * 100
        distance_matrix = np.round(np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2), 2).tolist()
        demands = [0] + [random.randint(1, 5) for _ in customers]
        vehicle_capacities = [random.randint(15, 30) for _ in range(num_vehicles)]
        vehicle_speeds = [random.randint(30, 60) for _ in range(num_vehicles)]
        vehicle_fixed_costs = [random.randint(0, 10) for _ in range(num_vehicles)]
        vehicle_cost_per_km = [round(random.uniform(0.005, 0.015), 2) for _ in range(num_vehicles)]
        time_windows = [(0, 100)]
        for _ in customers:
            e = random.randint(10, 30)
            l = e + random.randint(10, 30)
            time_windows.append((e, l))
        service_times = [0] + [random.randint(3, 10) for _ in customers]
        beta = 10
        gamma = 20
        eta = 10
        lambda_demand = [0] + [round(50 * demands[i], 2) for i in customers]
        cost_matrix = []
        for k in range(num_vehicles):
            cost_matrix.append([[round(distance_matrix[i][j] * vehicle_cost_per_km[k], 2) for j in nodes] for i in nodes])
        return {
            "num_customers": num_customers,
            "num_vehicles": num_vehicles,
            "depot": depot,
            "nodes": nodes,
            "customers": customers,
            "distance_matrix": distance_matrix,
            "demands": demands,
            "vehicle_capacities": vehicle_capacities,
            "vehicle_speeds": vehicle_speeds,
            "vehicle_fixed_costs": vehicle_fixed_costs,
            "vehicle_cost_per_km": vehicle_cost_per_km,
            "time_windows": time_windows,
            "service_times": service_times,
            "beta": beta,
            "gamma": gamma,
            "coords": coords.tolist(),
            "lambda_demand": lambda_demand,
            "eta": eta,
            "cost_matrix": cost_matrix,
            "w_fixed": 1.0,
            "w_distance": 1.0,
            "w_early": 1.0,
            "w_late": 1.0,
            "w_unserved": 1.0,
        }

    def _build_variables(self):
        model_data = self.model_data
        # x[i,j,k]: 是否车辆k从i到j
        for k in range(model_data["num_vehicles"]):
            for i in model_data["nodes"]:
                for j in model_data["nodes"]:
                    if i != j:
                        self.variable_names.append(f"x_{i}_{j}_{k}")
                        self.variable_types.append("I")
                        self.variable_bounds_lb.append(0)
                        self.variable_bounds_ub.append(1)
                        self.x_indices.append((i, j, k))
        # y[k]: 车辆k是否启用
        for k in range(model_data["num_vehicles"]):
            self.variable_names.append(f"y_{k}")
            self.variable_types.append("I")
            self.variable_bounds_lb.append(0)
            self.variable_bounds_ub.append(1)
            self.y_indices.append(k)
        # a[i]: 到达时间
        for i in model_data["nodes"]:
            self.variable_names.append(f"a_{i}")
            self.variable_types.append("C")
            self.variable_bounds_lb.append(0)
            self.variable_bounds_ub.append(1e5)
            self.a_indices.append(i)
        # u[i,k]: 车辆k在节点i的载重
        for i in model_data["nodes"]:
            for k in range(model_data["num_vehicles"]):
                self.variable_names.append(f"u_{i}_{k}")
                self.variable_types.append("C")
                self.variable_bounds_lb.append(0)
                self.variable_bounds_ub.append(model_data["vehicle_capacities"][k])
                self.u_indices.append((i, k))
        # delta_plus, delta_minus: 软时间窗惩罚
        for i in model_data["customers"]:
            self.variable_names.append(f"delta_plus_{i}")
            self.variable_types.append("C")
            self.variable_bounds_lb.append(0)
            self.variable_bounds_ub.append(1e5)
            self.delta_plus_indices.append(i)
            self.variable_names.append(f"delta_minus_{i}")
            self.variable_types.append("C")
            self.variable_bounds_lb.append(0)
            self.variable_bounds_ub.append(1e5)
            self.delta_minus_indices.append(i)
        # z[i]: 客户i是否被服务
        for i in model_data["customers"]:
            self.variable_names.append(f"z_{i}")
            self.variable_types.append("I")
            self.variable_bounds_lb.append(0)
            self.variable_bounds_ub.append(1)
            self.z_indices.append(i)

    def _build_problem_data(self):
        model_data = self.model_data
        variable_names = self.variable_names
        csr_offsets = [0]
        csr_indices = []
        csr_values = []
        upper_bounds = []
        lower_bounds = []
        M = 1e6

        # 约束：每辆车从depot出发一次
        for k in range(model_data["num_vehicles"]):
            row_indices = []
            row_values = []
            for j in model_data["nodes"]:
                if j != model_data["depot"]:
                    idx = variable_names.index(f"x_{model_data['depot']}_{j}_{k}")
                    row_indices.append(idx)
                    row_values.append(1.0)
            idy = variable_names.index(f"y_{k}")
            row_indices.append(idy)
            row_values.append(-1.0)
            csr_indices.extend(row_indices)
            csr_values.extend(row_values)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(0.0)
            lower_bounds.append(0.0)

        # 约束：每辆车回到depot一次
        for k in range(model_data["num_vehicles"]):
            row_indices = []
            row_values = []
            for j in model_data["nodes"]:
                if j != model_data["depot"]:
                    idx = variable_names.index(f"x_{j}_{model_data['depot']}_{k}")
                    row_indices.append(idx)
                    row_values.append(1.0)
            idy = variable_names.index(f"y_{k}")
            row_indices.append(idy)
            row_values.append(-1.0)
            csr_indices.extend(row_indices)
            csr_values.extend(row_values)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(0.0)
            lower_bounds.append(0.0)

        # 约束：每个客户被访问次数等于z_i
        for i in model_data["customers"]:
            row_indices = []
            row_values = []
            for k in range(model_data["num_vehicles"]):
                for j in model_data["nodes"]:
                    if j != i:
                        idx = variable_names.index(f"x_{j}_{i}_{k}")
                        row_indices.append(idx)
                        row_values.append(1.0)
            idx_z = variable_names.index(f"z_{i}")
            row_indices.append(idx_z)
            row_values.append(-1.0)
            csr_indices.extend(row_indices)
            csr_values.extend(row_values)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(0.0)
            lower_bounds.append(0.0)

        # 约束：每个节点流入等于流出
        for k in range(model_data["num_vehicles"]):
            for i in model_data["nodes"]:
                row_indices = []
                row_values = []
                for j in model_data["nodes"]:
                    if j != i:
                        idx_in = variable_names.index(f"x_{j}_{i}_{k}")
                        idx_out = variable_names.index(f"x_{i}_{j}_{k}")
                        row_indices.append(idx_in)
                        row_values.append(1.0)
                        row_indices.append(idx_out)
                        row_values.append(-1.0)
                csr_indices.extend(row_indices)
                csr_values.extend(row_values)
                csr_offsets.append(len(csr_indices))
                upper_bounds.append(0.0)
                lower_bounds.append(0.0)

        # 约束：车辆容量变化
        for k in range(model_data["num_vehicles"]):
            for i in model_data["nodes"]:
                for j in model_data["customers"]:
                    if j != i:
                        row_indices = []
                        row_values = []
                        idx_u_ik = variable_names.index(f"u_{i}_{k}")
                        row_indices.append(idx_u_ik)
                        row_values.append(1.0)
                        idx_u_jk = variable_names.index(f"u_{j}_{k}")
                        row_indices.append(idx_u_jk)
                        row_values.append(-1.0)
                        idx_z_j = variable_names.index(f"z_{j}")
                        row_indices.append(idx_z_j)
                        row_values.append(model_data["demands"][j])
                        idx_x_ijk = variable_names.index(f"x_{i}_{j}_{k}")
                        row_indices.append(idx_x_ijk)
                        row_values.append(model_data["vehicle_capacities"][k])
                        csr_indices.extend(row_indices)
                        csr_values.extend(row_values)
                        csr_offsets.append(len(csr_indices))
                        upper_bounds.append(model_data["vehicle_capacities"][k])
                        lower_bounds.append(-np.inf)
        for k in range(model_data["num_vehicles"]):
            for j in model_data["customers"]:
                row_indices = []
                row_values = []
                idx_u_jk = variable_names.index(f"u_{j}_{k}")
                row_indices.append(idx_u_jk)
                row_values.append(-1.0)
                idx_z_j = variable_names.index(f"z_{j}")
                row_indices.append(idx_z_j)
                row_values.append(model_data["demands"][j])
                csr_indices.extend(row_indices)
                csr_values.extend(row_values)
                csr_offsets.append(len(csr_indices))
                upper_bounds.append(0.0)
                lower_bounds.append(-np.inf)

        # 约束：时间窗与服务时间
        for k in range(model_data["num_vehicles"]):
            for i in model_data["nodes"]:
                for j in model_data["customers"]:
                    if i != j:
                        idx_a_i = variable_names.index(f"a_{i}")
                        idx_a_j = variable_names.index(f"a_{j}")
                        idx_x_ijk = variable_names.index(f"x_{i}_{j}_{k}")
                        idx_service_i = model_data["service_times"][i]
                        travel_time = model_data["distance_matrix"][i][j] / model_data["vehicle_speeds"][k]
                        row_indices = [idx_a_j, idx_a_i, idx_x_ijk]
                        row_values = [1.0, -1.0, -M]
                        csr_indices.extend(row_indices)
                        csr_values.extend(row_values)
                        csr_offsets.append(len(csr_indices))
                        upper_bounds.append(np.inf)
                        lower_bounds.append(idx_service_i + travel_time - M)

        for i in model_data["customers"]:
            row_indices = []
            row_values = []
            idx_delta_minus_i = variable_names.index(f"delta_minus_{i}")
            row_indices.append(idx_delta_minus_i)
            row_values.append(1.0)
            idx_a_i = variable_names.index(f"a_{i}")
            row_indices.append(idx_a_i)
            row_values.append(-1.0)
            csr_indices.extend(row_indices)
            csr_values.extend(row_values)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(np.inf)
            lower_bounds.append(-model_data["time_windows"][i][1])

        for i in model_data["customers"]:
            row_indices = []
            row_values = []
            idx_delta_plus_i = variable_names.index(f"delta_plus_{i}")
            row_indices.append(idx_delta_plus_i)
            row_values.append(1.0)
            idx_a_i = variable_names.index(f"a_{i}")
            row_indices.append(idx_a_i)
            row_values.append(1.0)
            csr_indices.extend(row_indices)
            csr_values.extend(row_values)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(np.inf)
            lower_bounds.append(model_data["time_windows"][i][0])

        # 约束：每辆车初始载重为0
        for k in range(model_data["num_vehicles"]):
            row_indices = []
            row_values = []
            idx_u_0k = variable_names.index(f"u_{model_data['depot']}_{k}")
            row_indices.append(idx_u_0k)
            row_values.append(1.0)
            csr_indices.extend(row_indices)
            csr_values.extend(row_values)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(0.0)
            lower_bounds.append(0.0)

        # 约束：depot到达时间为0
        idx_a_0 = variable_names.index(f"a_{model_data['depot']}")
        row_indices = [idx_a_0]
        row_values = [1.0]
        csr_indices.extend(row_indices)
        csr_values.extend(row_values)
        csr_offsets.append(len(csr_indices))
        upper_bounds.append(0.0)
        lower_bounds.append(0.0)

        # 约束：车辆启用约束（大M）
        for k in range(model_data["num_vehicles"]):
            row_indices = []
            row_values = []
            for i in model_data["nodes"]:
                for j in model_data["nodes"]:
                    if j != i:
                        idx = variable_names.index(f"x_{j}_{i}_{k}")
                        row_indices.append(idx)
                        row_values.append(1.0)
            idx_y = variable_names.index(f"y_{k}")
            row_indices.append(idx_y)
            row_values.append(-M)
            csr_indices.extend(row_indices)
            csr_values.extend(row_values)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(0.0)
            lower_bounds.append(-np.inf)

        # 目标函数
        objective_coeffs = [0.0] * len(variable_names)
        w_fixed = model_data.get("w_fixed", 1.0)
        w_distance = model_data.get("w_distance", 1.0)
        w_early = model_data.get("w_early", 1.0)
        w_late = model_data.get("w_late", 1.0)
        w_unserved = model_data.get("w_unserved", 1.0)

        for idx, k in enumerate(self.y_indices):
            objective_coeffs[variable_names.index(f"y_{k}")] += w_fixed * model_data["vehicle_fixed_costs"][k]
        for (i, j, k) in self.x_indices:
            idx = variable_names.index(f"x_{i}_{j}_{k}")
            objective_coeffs[idx] += w_distance * model_data["vehicle_cost_per_km"][k] * model_data["distance_matrix"][i][j]
        for i in model_data["customers"]:
            idx_plus = variable_names.index(f"delta_plus_{i}")
            idx_minus = variable_names.index(f"delta_minus_{i}")
            objective_coeffs[idx_plus] += w_early * model_data["beta"]
            objective_coeffs[idx_minus] += w_late * model_data["gamma"]
            idx_z = variable_names.index(f"z_{i}")
            objective_coeffs[idx_z] -= w_unserved * model_data["lambda_demand"][i]

        return {
            "csr_constraint_matrix": {
                "offsets": csr_offsets,
                "indices": csr_indices,
                "values": csr_values
            },
            "constraint_bounds": {
                "upper_bounds": upper_bounds,
                "lower_bounds": lower_bounds
            },
            "objective_data": {
                "coefficients": objective_coeffs,
                "scalability_factor": 1.0,
                "offset": 0.0
            },
            "variable_bounds": {
                "upper_bounds": self.variable_bounds_ub,
                "lower_bounds": self.variable_bounds_lb
            },
            "maximize": False,
            "variable_names": self.variable_names,
            "variable_types": self.variable_types,
            "solver_config": {
                "time_limit": 100
            }
        }

    def repoll(self, solution, repoll_tries=500):
        if "reqId" in solution and "response" not in solution:
            req_id = solution["reqId"]
            for _ in range(repoll_tries):
                solution = self.cuopt_service_client.repoll(req_id, response_type="dict")
                if "reqId" in solution and "response" in solution:
                    break
                time.sleep(1)
        return solution

    def solve(self):
        solution = self.cuopt_service_client.get_LP_solve(self.problem_data, response_type="dict")
        solution = self.repoll(solution)
        return solution

    def print_solution(self, solution):
        if "response" in solution:
            resp = solution["response"]
            if "solver_response" in resp and "solution" in resp["solver_response"]:
                print(json.dumps(resp, indent=4))
                sol = resp["solver_response"]["solution"]
                vars = sol.get("vars", {})
                objective = sol.get("primal_objective", None)
                for k in self.y_indices:
                    print(f"车辆{k}是否启用:", vars.get(f"y_{k}"))
                for i in self.a_indices:
                    print(f"客户{i}到达时间:", vars.get(f"a_{i}"),
                        "早到惩罚:", vars.get(f"delta_plus_{i}"),
                        "迟到惩罚:", vars.get(f"delta_minus_{i}"))
                print("目标值:", objective)
            else:
                print(json.dumps(resp, indent=4))
        else:
            print(json.dumps(solution, indent=4))

    def plot_routes(self, solution, tol=1e-4):
        """可视化车辆路径，tol为浮点误差容忍"""
        model_data = self.model_data
        coords = np.array(model_data["coords"])
        if "response" not in solution or "solver_response" not in solution["response"]:
            print("无可视化解")
            return
        sol = solution["response"]["solver_response"]["solution"]
        vars = sol.get("vars", {})
        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], c='blue', label='客户/仓库')
        plt.scatter(coords[0, 0], coords[0, 1], c='red', label='depot', marker='s')
        for k in self.y_indices:
            if vars.get(f"y_{k}", 0) < 0.5:
                continue
            # 构建路径
            route = [model_data["depot"]]
            current = model_data["depot"]
            visited = set([current])
            while True:
                found = False
                for j in model_data["nodes"]:
                    if j != current and abs(vars.get(f"x_{current}_{j}_{k}", 0.0) - 1.0) < tol and j not in visited:
                        plt.plot([coords[current, 0], coords[j, 0]], [coords[current, 1], coords[j, 1]], label=f'Vehicle {k}' if len(route)==1 else "", linewidth=2)
                        route.append(j)
                        visited.add(j)
                        current = j
                        found = True
                        break
                if not found:
                    # 回到depot
                    if abs(vars.get(f"x_{current}_{model_data['depot']}_{k}", 0.0) - 1.0) < tol:
                        plt.plot([coords[current, 0], coords[model_data['depot'], 0]], [coords[current, 1], coords[model_data['depot'], 1]], color='gray', linestyle='--')
                    break
        plt.legend()
        plt.title("VRP Solution Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

if __name__ == "__main__":
    solver = VRPMIPSolver()
    solution = solver.solve()
    solver.print_solution(solution)
    solver.plot_routes(solution)