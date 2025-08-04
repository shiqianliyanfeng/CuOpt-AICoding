import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pyscipopt

class VRPMIPBatchSolverSCIP:
    def __init__(self, data_path):
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

        model = pyscipopt.Model()
        model.setParam('display/verblevel', 0)
        model.setParam('limits/time', 30.0)  # 设置求解时间限制为30秒

        # Variables
        x = {}
        for k in range(num_vehicles):
            for i in nodes:
                for j in nodes:
                    if i != j:
                        x[i, j, k] = model.addVar(vtype="B", name=f"x_{i}_{j}_{k}")
        y = [model.addVar(vtype="B", name=f"y_{k}") for k in range(num_vehicles)]
        a = [model.addVar(lb=0, ub=None, vtype="C", name=f"a_{i}") for i in nodes]
        u = {}
        for i in nodes:
            for k in range(num_vehicles):
                u[i, k] = model.addVar(lb=0, ub=vehicle_capacities[k], vtype="C", name=f"u_{i}_{k}")
        delta_plus = [model.addVar(lb=0, ub=None, vtype="C", name=f"delta_plus_{i}") for i in customers]
        delta_minus = [model.addVar(lb=0, ub=None, vtype="C", name=f"delta_minus_{i}") for i in customers]
        z = [model.addVar(vtype="B", name=f"z_{i}") for i in customers]

        # Constraints

        # 每辆车从depot出发一次
        for k in range(num_vehicles):
            model.addCons(
                pyscipopt.quicksum(x[depot, j, k] for j in nodes if j != depot) == y[k]
            )
        # 每辆车回到depot一次
        for k in range(num_vehicles):
            model.addCons(
                pyscipopt.quicksum(x[j, depot, k] for j in nodes if j != depot) == y[k]
            )
        # 每个客户被访问次数等于z_i
        for idx, i in enumerate(customers):
            model.addCons(
                pyscipopt.quicksum(x[j, i, k] for k in range(num_vehicles) for j in nodes if j != i) == z[idx]
            )
        # 每个节点流入等于流出
        for k in range(num_vehicles):
            for i in nodes:
                model.addCons(
                    pyscipopt.quicksum(x[j, i, k] for j in nodes if j != i) ==
                    pyscipopt.quicksum(x[i, j, k] for j in nodes if j != i)
                )
        # 车辆容量变化
        for k in range(num_vehicles):
            for i in nodes:
                for idx, j in enumerate(customers):
                    if j != i:
                        model.addCons(
                            u[i, k] - u[j, k] + demands[j] * z[idx] + vehicle_capacities[k] * x[i, j, k] <= vehicle_capacities[k]
                        )
        for k in range(num_vehicles):
            for idx, j in enumerate(customers):
                model.addCons(
                    -u[j, k] + demands[j] * z[idx] <= 0
                )
        # 时间窗与服务时间
        M = 1e5
        for k in range(num_vehicles):
            for i in nodes:
                for idx, j in enumerate(customers):
                    if i != j:
                        travel_time = distance_matrix[i][j] / vehicle_speeds[k]
                        model.addCons(
                            a[j] - a[i] >= service_times[i] + travel_time - M * (1 - x[i, j, k])
                        )
        for idx, i in enumerate(customers):
            model.addCons(delta_minus[idx] + a[i] >= time_windows[i][1])
            model.addCons(delta_plus[idx] + a[i] >= time_windows[i][0])
        # 每辆车初始载重为0
        for k in range(num_vehicles):
            model.addCons(u[depot, k] == 0)
        # depot到达时间为0
        model.addCons(a[depot] == 0)

        # 车辆启用约束（大M）
        for k in range(num_vehicles):
            model.addCons(
                pyscipopt.quicksum(x[j, i, k] for i in nodes for j in nodes if j != i) <= len(nodes) * y[k]
            )

        # 目标函数
        obj = (
            w_fixed * pyscipopt.quicksum(vehicle_fixed_costs[k] * y[k] for k in range(num_vehicles)) +
            w_distance * pyscipopt.quicksum(vehicle_cost_per_km[k] * distance_matrix[i][j] * x[i, j, k]
                             for k in range(num_vehicles) for i in nodes for j in nodes if i != j) +
            w_early * beta * pyscipopt.quicksum(delta_plus[idx] for idx in range(len(customers))) +
            w_late * gamma * pyscipopt.quicksum(delta_minus[idx] for idx in range(len(customers))) -
            w_unserved * pyscipopt.quicksum(lambda_demand[i] * z[idx] for idx, i in enumerate(customers))
        )
        model.setObjective(obj, "minimize")

        # Solve
        start = time.time()
        model.optimize()
        elapsed = time.time() - start

        status = model.getStatus()
        if status in ["optimal", "bestsollimit", "timelimit"]:
            objective = model.getObjVal()
            used_vehicles = sum(1 for k in range(num_vehicles) if model.getVal(y[k]) > 0.5)
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
    solver = VRPMIPBatchSolverSCIP("../data/vrp_dataset_100.json")
    solver.batch_solve(max_instances=10)
    solver.show_stats()