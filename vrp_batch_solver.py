import json
import numpy as np
import cuopt
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

    def solve_mip(self, instance):
        # MIP方式建模（使用cuOpt Model接口，简化版）
        model = cuopt.Model()
        nodes = instance["nodes"]
        customers = instance["customers"]
        num_vehicles = instance["num_vehicles"]
        vehicle_capacities = instance["vehicle_capacities"]
        vehicle_fixed_costs = instance["vehicle_fixed_costs"]
        vehicle_speeds = instance["vehicle_speeds"]
        vehicle_cost_per_km = instance["vehicle_cost_per_km"]
        distance_matrix = np.array(instance["distance_matrix"])
        demands = instance["demands"]
        time_windows = instance["time_windows"]
        service_times = instance["service_times"]
        beta = instance["beta"]
        gamma = instance["gamma"]

        # 决策变量
        x = {}
        for k in range(num_vehicles):
            for i in nodes:
                for j in nodes:
                    if i != j:
                        x[i, j, k] = model.add_variable(name=f"x_{i}_{j}_{k}", var_type="binary")
        y = {k: model.add_variable(name=f"y_{k}", var_type="binary") for k in range(num_vehicles)}
        a = {i: model.add_variable(name=f"a_{i}", lb=0) for i in nodes}
        u = {(i, k): model.add_variable(name=f"u_{i}_{k}", lb=0, ub=vehicle_capacities[k]) for i in nodes for k in range(num_vehicles)}
        delta_plus = {i: model.add_variable(name=f"delta_plus_{i}", lb=0) for i in customers}
        delta_minus = {i: model.add_variable(name=f"delta_minus_{i}", lb=0) for i in customers}

        # 目标函数
        fixed_cost = sum(vehicle_fixed_costs[k] * y[k] for k in range(num_vehicles))
        transport_cost = sum(
            vehicle_cost_per_km[k] * distance_matrix[i][j] * x[i, j, k]
            for k in range(num_vehicles) for i in nodes for j in nodes if i != j
        )
        time_window_penalty = sum(
            beta * delta_plus[i] + gamma * delta_minus[i] for i in customers
        )
        model.set_objective(fixed_cost + transport_cost + time_window_penalty, sense="min")

        # 约束（简化版，详见model.txt）
        depot = 0
        # 每个客户仅被服务一次
        for i in customers:
            model.add_constraint(
                sum(x[j, i, k] for k in range(num_vehicles) for j in nodes if j != i) == 1
            )
        # 流量守恒
        for k in range(num_vehicles):
            model.add_constraint(
                sum(x[depot, j, k] for j in customers) == y[k]
            )
            model.add_constraint(
                sum(x[j, depot, k] for j in customers) == y[k]
            )
        # 容量约束
        for k in range(num_vehicles):
            for i in customers:
                model.add_constraint(u[i, k] >= demands[i])
                model.add_constraint(u[i, k] <= vehicle_capacities[k])
            for i in customers:
                for j in customers:
                    if i != j:
                        model.add_constraint(
                            u[i, k] + demands[j] - u[j, k] <= vehicle_capacities[k] * (1 - x[i, j, k])
                        )
        # 时间窗约束
        M = 1e5
        for k in range(num_vehicles):
            for i in nodes:
                for j in customers:
                    if i != j:
                        travel_time = distance_matrix[i][j] / vehicle_speeds[k]
                        model.add_constraint(
                            a[j] >= a[i] + service_times[i] + travel_time - M * (1 - x[i, j, k])
                        )
        for i in customers:
            e_i, l_i = time_windows[i]
            model.add_constraint(delta_plus[i] >= e_i - a[i])
            model.add_constraint(delta_minus[i] >= a[i] - l_i)
        # 车辆启用关联
        for k in range(num_vehicles):
            model.add_constraint(
                sum(x[i, j, k] for i in nodes for j in nodes if i != j) <= M * y[k]
            )

        # 求解
        start = time.time()
        solution = model.solve()
        elapsed = time.time() - start

        used_vehicles = sum(solution[f"y_{k}"] for k in range(num_vehicles))
        return solution["objective_value"], elapsed, used_vehicles

    def solve_vrp(self, instance):
        # VRP方式建模（cuOpt Solver接口）
        solver = cuopt.Solver()
        solver.set_distance_matrix(np.array(instance["distance_matrix"]))
        solver.set_vehicle_capacity(instance["vehicle_capacities"])
        solver.set_vehicle_fixed_cost(instance["vehicle_fixed_costs"])
        solver.set_vehicle_speed(instance["vehicle_speeds"])
        solver.set_vehicle_cost_per_distance(instance["vehicle_cost_per_km"])
        solver.set_demands(instance["demands"])
        solver.set_time_windows(instance["time_windows"])
        solver.set_service_times(instance["service_times"])
        solver.set_time_window_penalty(instance["beta"], instance["gamma"])

        start = time.time()
        result = solver.solve()
        elapsed = time.time() - start

        used_vehicles = sum(1 for route in result.routes if len(route) > 2)
        return result.objective, elapsed, used_vehicles, result.routes

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
            print(f"{method.upper()} Instance {idx+1}/{self.num_instances}: Obj={obj:.2f}, Time={t:.2f}s, Used Vehicles={used}")

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

    def visualize_progress(self, method="vrp", instance_idx=0, interval=0.5, max_time=10):
        """
        可视化单次求解过程中当前时间最好目标值的变化曲线。
        仅支持cuOpt Solver（VRP方式），MIP方式需自行实现callback。
        """
        instance = self.data[instance_idx]
        solver = cuopt.Solver()
        solver.set_distance_matrix(np.array(instance["distance_matrix"]))
        solver.set_vehicle_capacity(instance["vehicle_capacities"])
        solver.set_vehicle_fixed_cost(instance["vehicle_fixed_costs"])
        solver.set_vehicle_speed(instance["vehicle_speeds"])
        solver.set_vehicle_cost_per_distance(instance["vehicle_cost_per_km"])
        solver.set_demands(instance["demands"])
        solver.set_time_windows(instance["time_windows"])
        solver.set_service_times(instance["service_times"])
        solver.set_time_window_penalty(instance["beta"], instance["gamma"])

        best_objectives = []
        times = []
        start = time.time()
        elapsed = 0

        # cuOpt Solver支持callback或progress接口时可用，否则模拟
        while elapsed < max_time:
            result = solver.solve(time_limit=interval)
            elapsed = time.time() - start
            best_objectives.append(result.objective)
            times.append(elapsed)
            if elapsed + interval > max_time:
                break

        plt.figure(figsize=(8,5))
        plt.plot(times, best_objectives, marker='o')
        plt.xlabel("Time (s)")
        plt.ylabel("Best Objective Value")
        plt.title(f"求解进程目标值变化 (Instance {instance_idx}, {method.upper()})")
        plt.grid(True)
        plt.show()

# 用法示例
if __name__ == "__main__":
    solver = VRPBatchSolver("vrp_dataset_100.json")
    solver.batch_solve(method="vrp")
    solver.batch_solve(method="mip")
    solver.show_stats()
    solver.plot_stats()
    solver.plot_routes(instance_idx=0, method="vrp")
    solver.visualize_progress(method="vrp", instance_idx=0, interval=1, max_time=10)