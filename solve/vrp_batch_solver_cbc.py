import json
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from ortools.linear_solver import pywraplp

class VRPMIPBatchSolverCBC:
    def __init__(self, data_path):
        print(os.getcwd())
        with open(data_path, "r") as f:
            self.instances = json.load(f)
        self.num_instances = len(self.instances)
        self.stats = {
            "objective": [],
            "time": [],
            "used_vehicles": []
        }

    def solve_instance(self, instance):
        num_vehicles = instance["num_vehicles"]
        nodes = instance["nodes"]
        customers = instance["customers"]
        depot = instance["depot"]
        demands = instance["demands"]
        vehicle_capacities = instance["vehicle_capacities"]
        distance_matrix = np.array(instance["distance_matrix"])
        vehicle_fixed_costs = instance["vehicle_fixed_costs"]
        vehicle_cost_per_km = instance["vehicle_cost_per_km"]
        time_windows = instance["time_windows"]
        service_times = instance["service_times"]
        vehicle_speeds = instance["vehicle_speeds"]
        beta = instance.get("beta", 10)
        gamma = instance.get("gamma", 20)
        lambda_demand = instance.get("lambda_demand", [0] * len(nodes))
        w_fixed = instance.get("w_fixed", 1.0)
        w_distance = instance.get("w_distance", 1.0)
        w_early = instance.get("w_early", 1.0)
        w_late = instance.get("w_late", 1.0)
        w_unserved = instance.get("w_unserved", 1.0)

        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            raise RuntimeError("CBC solver not available.")

        # 设置求解时间限制为30秒
        solver.SetTimeLimit(30000)  # 单位为毫秒

        # Variables
        x = {}
        for k in range(num_vehicles):
            for i in nodes:
                for j in nodes:
                    if i != j:
                        x[i, j, k] = solver.BoolVar(f"x_{i}_{j}_{k}")
        y = [solver.BoolVar(f"y_{k}") for k in range(num_vehicles)]
        a = [solver.NumVar(0, solver.infinity(), f"a_{i}") for i in nodes]
        u = {}
        for i in nodes:
            for k in range(num_vehicles):
                u[i, k] = solver.NumVar(0, vehicle_capacities[k], f"u_{i}_{k}")
        delta_plus = [solver.NumVar(0, solver.infinity(), f"delta_plus_{i}") for i in customers]
        delta_minus = [solver.NumVar(0, solver.infinity(), f"delta_minus_{i}") for i in customers]
        z = [solver.BoolVar(f"z_{i}") for i in customers]

        # Constraints

        # 每辆车从depot出发一次
        for k in range(num_vehicles):
            solver.Add(
                sum(x[depot, j, k] for j in nodes if j != depot) == y[k]
            )
        # 每辆车回到depot一次
        for k in range(num_vehicles):
            solver.Add(
                sum(x[j, depot, k] for j in nodes if j != depot) == y[k]
            )
        # 每个客户被访问次数等于z_i
        for idx, i in enumerate(customers):
            solver.Add(
                sum(x[j, i, k] for k in range(num_vehicles) for j in nodes if j != i) == z[idx]
            )
        # 每个节点流入等于流出
        for k in range(num_vehicles):
            for i in nodes:
                solver.Add(
                    sum(x[j, i, k] for j in nodes if j != i) == sum(x[i, j, k] for j in nodes if j != i)
                )
        # 车辆容量变化
        for k in range(num_vehicles):
            for i in nodes:
                for idx, j in enumerate(customers):
                    if j != i:
                        solver.Add(
                            u[i, k] - u[j, k] + demands[j] * z[idx] + vehicle_capacities[k] * x[i, j, k] <= vehicle_capacities[k]
                        )
        for k in range(num_vehicles):
            for idx, j in enumerate(customers):
                solver.Add(
                    -u[j, k] + demands[j] * z[idx] <= 0
                )
        # 时间窗与服务时间
        M = 1e5
        for k in range(num_vehicles):
            for i in nodes:
                for idx, j in enumerate(customers):
                    if i != j:
                        travel_time = distance_matrix[i][j] / vehicle_speeds[k]
                        solver.Add(
                            a[j] - a[i] >= service_times[i] + travel_time - M * (1 - x[i, j, k])
                        )
        for idx, i in enumerate(customers):
            solver.Add(delta_minus[idx] + a[i] >= time_windows[i][1])
            solver.Add(delta_plus[idx] + a[i] >= time_windows[i][0])
        # 每辆车初始载重为0
        for k in range(num_vehicles):
            solver.Add(u[depot, k] == 0)
        # depot到达时间为0
        solver.Add(a[depot] == 0)

        # 车辆启用约束（大M）
        for k in range(num_vehicles):
            solver.Add(
                sum(x[j, i, k] for i in nodes for j in nodes if j != i) <= len(nodes) * y[k]
            )

        # 目标函数
        obj = (
            w_fixed * sum(vehicle_fixed_costs[k] * y[k] for k in range(num_vehicles)) +
            w_distance * sum(vehicle_cost_per_km[k] * distance_matrix[i][j] * x[i, j, k]
                             for k in range(num_vehicles) for i in nodes for j in nodes if i != j) +
            w_early * beta * sum(delta_plus[idx] for idx in range(len(customers))) +
            w_late * gamma * sum(delta_minus[idx] for idx in range(len(customers))) -
            w_unserved * sum(lambda_demand[i] * z[idx] for idx, i in enumerate(customers))
        )
        solver.Minimize(obj)

        # Solve
        start = time.time()
        status = solver.Solve()
        elapsed = time.time() - start

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            objective = solver.Objective().Value()
            used_vehicles = sum(1 for k in range(num_vehicles) if y[k].solution_value() > 0.5)
        else:
            objective = None
            used_vehicles = 0

        return objective, elapsed, used_vehicles

    def batch_solve(self, max_instances=10):
        for idx, instance in enumerate(self.instances):
            if idx < max_instances:
                obj, t, used = self.solve_instance(instance)
                self.stats["objective"].append(obj)
                self.stats["time"].append(t)
                self.stats["used_vehicles"].append(used)
                print(f"Instance {idx+1}/{self.num_instances}: Obj={obj}, Time={t:.2f}s, Used Vehicles={used}")

    def show_stats(self):
        if self.stats["objective"]:
            print(f"\n统计：")
            print(f"平均目标值: {np.mean([x for x in self.stats['objective'] if x is not None]):.2f}")
            print(f"平均用时: {np.mean(self.stats['time']):.2f}s")
            print(f"平均车辆数: {np.mean(self.stats['used_vehicles']):.2f}")

    def plot_stats(self):
        plt.figure(figsize=(12,4))
        metrics = ["objective", "time", "used_vehicles"]
        titles = ["Objective", "Time (s)", "Used Vehicles"]
        for i, metric in enumerate(metrics):
            plt.subplot(1,3,i+1)
            plt.plot(self.stats[metric], marker='o')
            plt.title(titles[i])
            plt.xlabel("Instance")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    solver = VRPMIPBatchSolverCBC("../data/vrp_dataset_100.json")
    solver.batch_solve(max_instances=10)
    solver.show_stats()