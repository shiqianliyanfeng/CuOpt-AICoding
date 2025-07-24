from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import numpy as np
import matplotlib.pyplot as plt
import time

class VRPBatchSolver:
    def __init__(self, data_path):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.num_instances = len(self.data)
        self.stats = {
            "mip": {"objective": [], "time": [], "used_vehicles": []},
            "vrp": {"objective": [], "time": [], "used_vehicles": []}
        }
        self.cuopt_service_client = CuOptServiceSelfHostClient(
            ip="0.0.0.0",
            port=8000,
            polling_timeout=25,
            timeout_exception=False
        )

    def repoll(self, solution, repoll_tries=500):
        if "reqId" in solution and "response" not in solution:
            req_id = solution["reqId"]
            for i in range(repoll_tries):
                solution = self.cuopt_service_client.repoll(req_id, response_type="dict")
                if "reqId" in solution and "response" in solution:
                    break
                time.sleep(1)
        return solution

    def solve_vrp(self, instance):
        # 构造 routing server 所需数据结构
        num_vehicles = instance["num_vehicles"]
        depot = instance["depot"] if "depot" in instance else 0
        nodes = instance["nodes"]
        customers = instance["customers"]
        distance_matrix = np.array(instance["distance_matrix"])
        demands = instance["demands"]
        vehicle_capacities = instance["vehicle_capacities"]
        vehicle_fixed_costs = instance["vehicle_fixed_costs"]
        vehicle_speeds = instance["vehicle_speeds"]
        vehicle_cost_per_km = instance["vehicle_cost_per_km"]
        time_windows = instance["time_windows"]
        service_times = instance["service_times"]
        beta = instance.get("beta", 10)
        gamma = instance.get("gamma", 20)

        data = {
            "cost_matrix_data": {
                "data": {
                    "0": distance_matrix.tolist()
                }
            },
            "task_data": {
                "task_locations": customers,
                "demands": [demands[i] for i in customers],
                "time_windows": [list(time_windows[i]) for i in customers],
                "service_times": [service_times[i] for i in customers]
            },
            "fleet_data": {
                "vehicle_locations": [[depot] for _ in range(num_vehicles)],
                "capacities": vehicle_capacities,
                "fixed_costs": vehicle_fixed_costs,
                "speeds": vehicle_speeds,
                "cost_per_distance": vehicle_cost_per_km,
                "time_windows": [list(time_windows[depot]) for _ in range(num_vehicles)]
            },
            "penalties": {
                "early": beta,
                "late": gamma
            }
        }

        start = time.time()
        solution = self.cuopt_service_client.get_optimized_routes(data)
        solution = self.repoll(solution)
        elapsed = time.time() - start

        used_vehicles = 0
        objective = None
        routes = []
        if "response" in solution:
            resp = solution["response"]
            objective = resp.get("objective_value", None)
            routes = resp.get("routes", [])
            used_vehicles = sum(1 for route in routes if len(route) > 2)
        return objective, elapsed, used_vehicles, routes

    def solve_mip(self, instance):
        # 构造MIP问题CSR格式数据（仅举例，完整约束需补充）
        num_vehicles = instance["num_vehicles"]
        depot = instance["depot"] if "depot" in instance else 0
        nodes = instance["nodes"]
        customers = instance["customers"]
        distance_matrix = np.array(instance["distance_matrix"])
        demands = instance["demands"]
        vehicle_capacities = instance["vehicle_capacities"]
        vehicle_fixed_costs = instance["vehicle_fixed_costs"]
        vehicle_speeds = instance["vehicle_speeds"]
        vehicle_cost_per_km = instance["vehicle_cost_per_km"]
        time_windows = instance["time_windows"]
        service_times = instance["service_times"]
        beta = instance.get("beta", 10)
        gamma = instance.get("gamma", 20)

        variable_names = []
        variable_types = []
        variable_bounds_lb = []
        variable_bounds_ub = []

        x_indices = []
        for k in range(num_vehicles):
            for i in nodes:
                for j in nodes:
                    if i != j:
                        variable_names.append(f"x_{i}_{j}_{k}")
                        variable_types.append("B")
                        variable_bounds_lb.append(0)
                        variable_bounds_ub.append(1)
                        x_indices.append((i, j, k))
        y_indices = []
        for k in range(num_vehicles):
            variable_names.append(f"y_{k}")
            variable_types.append("B")
            variable_bounds_lb.append(0)
            variable_bounds_ub.append(1)
            y_indices.append(k)
        a_indices = []
        for i in nodes:
            variable_names.append(f"a_{i}")
            variable_types.append("C")
            variable_bounds_lb.append(0)
            variable_bounds_ub.append(1e5)
            a_indices.append(i)
        u_indices = []
        for i in nodes:
            for k in range(num_vehicles):
                variable_names.append(f"u_{i}_{k}")
                variable_types.append("C")
                variable_bounds_lb.append(0)
                variable_bounds_ub.append(vehicle_capacities[k])
                u_indices.append((i, k))
        delta_plus_indices = []
        delta_minus_indices = []
        for i in customers:
            variable_names.append(f"delta_plus_{i}")
            variable_types.append("C")
            variable_bounds_lb.append(0)
            variable_bounds_ub.append(1e5)
            delta_plus_indices.append(i)
            variable_names.append(f"delta_minus_{i}")
            variable_types.append("C")
            variable_bounds_lb.append(0)
            variable_bounds_ub.append(1e5)
            delta_minus_indices.append(i)

        # 约束和目标函数转CSR格式（这里只写部分约束，完整模型需补充所有约束）
        csr_offsets = [0]
        csr_indices = []
        csr_values = []
        upper_bounds = []
        lower_bounds = []

        # 每个客户仅被服务一次
        for i in customers:
            row_indices = []
            row_values = []
            for k in range(num_vehicles):
                for j in nodes:
                    if j != i:
                        idx = variable_names.index(f"x_{j}_{i}_{k}")
                        row_indices.append(idx)
                        row_values.append(1.0)
            csr_indices.extend(row_indices)
            csr_values.extend(row_values)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(1.0)
            lower_bounds.append(1.0)

        # 目标函数
        objective_coeffs = [0.0] * len(variable_names)
        for k in y_indices:
            objective_coeffs[variable_names.index(f"y_{k}")] += vehicle_fixed_costs[k]
        for (i, j, k) in x_indices:
            idx = variable_names.index(f"x_{i}_{j}_{k}")
            objective_coeffs[idx] += vehicle_cost_per_km[k] * distance_matrix[i][j]
        for i in customers:
            idx_plus = variable_names.index(f"delta_plus_{i}")
            idx_minus = variable_names.index(f"delta_minus_{i}")
            objective_coeffs[idx_plus] += beta
            objective_coeffs[idx_minus] += gamma

        data = {
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
                "upper_bounds": variable_bounds_ub,
                "lower_bounds": variable_bounds_lb
            },
            "maximize": False,
            "variable_names": variable_names,
            "variable_types": variable_types,
            "solver_config": {
                "time_limit": 60
            }
        }

        start = time.time()
        solution = self.cuopt_service_client.get_LP_solve(data, response_type="dict")
        solution = self.repoll(solution)
        elapsed = time.time() - start

        used_vehicles = 0
        objective = None
        if "response" in solution:
            resp = solution["response"]
            objective = resp.get("objective_value", None)
            used_vehicles = sum(resp["variable_values"][variable_names.index(f"y_{k}")] for k in y_indices)
        return objective, elapsed, used_vehicles

    def batch_solve(self, method="mip"):
        for idx, instance in enumerate(self.data):
            if method == "mip":
                obj, t, used = self.solve_mip(instance)
                self.stats["mip"]["objective"].append(obj)
                self.stats["mip"]["time"].append(t)
                self.stats["mip"]["used_vehicles"].append(used)
            elif method == "vrp":
                obj, t, used, _ = self.solve_vrp(instance)
                self.stats["vrp"]["objective"].append(obj)
                self.stats["vrp"]["time"].append(t)
                self.stats["vrp"]["used_vehicles"].append(used)
            print(f"{method.upper()} Instance {idx+1}/{self.num_instances}: Obj={obj}, Time={t:.2f}s, Used Vehicles={used}")

    def show_stats(self):
        for method in ["mip", "vrp"]:
            if self.stats[method]["objective"]:
                print(f"\n{method.upper()}统计：")
                print(f"平均目标值: {np.mean(self.stats[method]['objective']):.2f}")
                print(f"平均用时: {np.mean(self.stats[method]['time']):.2f}s")
                print(f"平均车辆数: {np.mean(self.stats[method]['used_vehicles']):.2f}")

    def plot_stats(self):
        plt.figure(figsize=(12,5))
        for i, metric in enumerate(["objective", "time", "used_vehicles"]):
            plt.subplot(1,3,i+1)
            for method in ["mip", "vrp"]:
                if self.stats[method][metric]:
                    plt.plot(self.stats[method][metric], label=method)
            plt.title(metric)
            plt.xlabel("Instance")
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_routes(self, instance_idx=0, method="vrp"):
        instance = self.data[instance_idx]
        coords = np.random.rand(len(instance["nodes"]), 2) * 100  # 若有坐标可替换
        if method == "vrp":
            _, _, _, routes = self.solve_vrp(instance)
        else:
            print("MIP方式暂不支持路径可视化")
            return
        plt.figure(figsize=(8,8))
        for k, route in enumerate(routes):
            route_coords = coords[route]
            plt.plot(route_coords[:,0], route_coords[:,1], marker='o', label=f"Vehicle {k}")
        plt.scatter(coords[0,0], coords[0,1], c='r', marker='s', s=100, label="Depot")
        plt.title(f"Routes for Instance {instance_idx} ({method.upper()})")
        plt.legend()
        plt.show()

# 用法示例
if __name__ == "__main__":
    solver = VRPBatchSolver("vrp_dataset_100.json")
    solver.batch_solve(method="vrp")
    solver.batch_solve(method="mip")
    solver.show_stats()
    solver.plot_stats()
    solver.plot_routes(instance_idx=0, method="vrp")