from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import os

# callback should accept 2 values, one is solution and another is cost
def callback(solution, solution_cost):
    print(f"Solution : {solution} cost : {solution_cost}\n")

# Logging callback
def log_callback(log):
    for i in log:
        print("server-log: ", i)

class VRPMIPBatchSolver:
    def __init__(self, data_path, ip="0.0.0.0", port=5001):
        with open(data_path, "r") as f:
            self.instances = json.load(f)
        self.num_instances = len(self.instances)
        self.stats = {
            "objective": [],
            "time": [],
            "used_vehicles": []
        }
        self.cuopt_service_client = CuOptServiceSelfHostClient(
            ip=ip,
            port=port,
            polling_timeout=25,
            timeout_exception=False
        )

    def repoll(self, solution, repoll_tries=500):
        if "reqId" in solution and "response" not in solution:
            req_id = solution["reqId"]
            for _ in range(repoll_tries):
                solution = self.cuopt_service_client.repoll(req_id, response_type="dict")
                if "reqId" in solution and "response" in solution:
                    break
                time.sleep(1)
        return solution

    def build_problem_data(self, instance):
        # 参考 vrp_solver.py 的建模方式
        variable_names = []
        variable_types = []
        variable_bounds_lb = []
        variable_bounds_ub = []
        x_indices = []
        y_indices = []
        a_indices = []
        u_indices = []
        delta_plus_indices = []
        delta_minus_indices = []
        z_indices = []

        num_vehicles = instance["num_vehicles"]
        nodes = instance["nodes"]
        customers = instance["customers"]

        # x[i,j,k]
        for k in range(num_vehicles):
            for i in nodes:
                for j in nodes:
                    if i != j:
                        variable_names.append(f"x_{i}_{j}_{k}")
                        variable_types.append("I")
                        variable_bounds_lb.append(0)
                        variable_bounds_ub.append(1)
                        x_indices.append((i, j, k))
        # y[k]
        for k in range(num_vehicles):
            variable_names.append(f"y_{k}")
            variable_types.append("I")
            variable_bounds_lb.append(0)
            variable_bounds_ub.append(1)
            y_indices.append(k)
        # a[i]
        for i in nodes:
            variable_names.append(f"a_{i}")
            variable_types.append("C")
            variable_bounds_lb.append(0)
            variable_bounds_ub.append(1e5)
            a_indices.append(i)
        # u[i,k]
        for i in nodes:
            for k in range(num_vehicles):
                variable_names.append(f"u_{i}_{k}")
                variable_types.append("C")
                variable_bounds_lb.append(0)
                variable_bounds_ub.append(instance["vehicle_capacities"][k])
                u_indices.append((i, k))
        # delta_plus, delta_minus
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
        # z[i]
        for i in customers:
            variable_names.append(f"z_{i}")
            variable_types.append("I")
            variable_bounds_lb.append(0)
            variable_bounds_ub.append(1)
            z_indices.append(i)

        # 约束和目标函数
        csr_offsets = [0]
        csr_indices = []
        csr_values = []
        upper_bounds = []
        lower_bounds = []
        M = 1e6

        # 约束：每辆车从depot出发一次
        for k in range(num_vehicles):
            row_indices = []
            row_values = []
            for j in nodes:
                if j != instance["depot"]:
                    idx = variable_names.index(f"x_{instance['depot']}_{j}_{k}")
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
        for k in range(num_vehicles):
            row_indices = []
            row_values = []
            for j in nodes:
                if j != instance["depot"]:
                    idx = variable_names.index(f"x_{j}_{instance['depot']}_{k}")
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
        for i in customers:
            row_indices = []
            row_values = []
            for k in range(num_vehicles):
                for j in nodes:
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
        for k in range(num_vehicles):
            for i in nodes:
                row_indices = []
                row_values = []
                for j in nodes:
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
        for k in range(num_vehicles):
            for i in nodes:
                for j in customers:
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
                        row_values.append(instance["demands"][j])
                        idx_x_ijk = variable_names.index(f"x_{i}_{j}_{k}")
                        row_indices.append(idx_x_ijk)
                        row_values.append(instance["vehicle_capacities"][k])
                        csr_indices.extend(row_indices)
                        csr_values.extend(row_values)
                        csr_offsets.append(len(csr_indices))
                        upper_bounds.append(instance["vehicle_capacities"][k])
                        lower_bounds.append(-np.inf)
        for k in range(num_vehicles):
            for j in customers:
                row_indices = []
                row_values = []
                idx_u_jk = variable_names.index(f"u_{j}_{k}")
                row_indices.append(idx_u_jk)
                row_values.append(-1.0)
                idx_z_j = variable_names.index(f"z_{j}")
                row_indices.append(idx_z_j)
                row_values.append(instance["demands"][j])
                csr_indices.extend(row_indices)
                csr_values.extend(row_values)
                csr_offsets.append(len(csr_indices))
                upper_bounds.append(0.0)
                lower_bounds.append(-np.inf)

        # 约束：时间窗与服务时间
        for k in range(num_vehicles):
            for i in nodes:
                for j in customers:
                    if i != j:
                        idx_a_i = variable_names.index(f"a_{i}")
                        idx_a_j = variable_names.index(f"a_{j}")
                        idx_x_ijk = variable_names.index(f"x_{i}_{j}_{k}")
                        idx_service_i = instance["service_times"][i]
                        travel_time = instance["distance_matrix"][i][j] / instance["vehicle_speeds"][k]
                        row_indices = [idx_a_j, idx_a_i, idx_x_ijk]
                        row_values = [1.0, -1.0, -M]
                        csr_indices.extend(row_indices)
                        csr_values.extend(row_values)
                        csr_offsets.append(len(csr_indices))
                        upper_bounds.append(np.inf)
                        lower_bounds.append(idx_service_i + travel_time - M)

        for i in customers:
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
            lower_bounds.append(-instance["time_windows"][i][1])

        for i in customers:
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
            lower_bounds.append(instance["time_windows"][i][0])

        # 约束：每辆车初始载重为0
        for k in range(num_vehicles):
            row_indices = []
            row_values = []
            idx_u_0k = variable_names.index(f"u_{instance['depot']}_{k}")
            row_indices.append(idx_u_0k)
            row_values.append(1.0)
            csr_indices.extend(row_indices)
            csr_values.extend(row_values)
            csr_offsets.append(len(csr_indices))
            upper_bounds.append(0.0)
            lower_bounds.append(0.0)

        # 约束：depot到达时间为0
        idx_a_0 = variable_names.index(f"a_{instance['depot']}")
        row_indices = [idx_a_0]
        row_values = [1.0]
        csr_indices.extend(row_indices)
        csr_values.extend(row_values)
        csr_offsets.append(len(csr_indices))
        upper_bounds.append(0.0)
        lower_bounds.append(0.0)

        # 约束：车辆启用约束（大M）
        for k in range(num_vehicles):
            row_indices = []
            row_values = []
            for i in nodes:
                for j in nodes:
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
        w_fixed = instance.get("w_fixed", 1.0)
        w_distance = instance.get("w_distance", 1.0)
        w_early = instance.get("w_early", 1.0)
        w_late = instance.get("w_late", 1.0)
        w_unserved = instance.get("w_unserved", 1.0)

        for idx, k in enumerate(y_indices):
            objective_coeffs[variable_names.index(f"y_{k}")] += w_fixed * instance["vehicle_fixed_costs"][k]
        for (i, j, k) in x_indices:
            idx = variable_names.index(f"x_{i}_{j}_{k}")
            objective_coeffs[idx] += w_distance * instance["vehicle_cost_per_km"][k] * instance["distance_matrix"][i][j]
        for i in customers:
            idx_plus = variable_names.index(f"delta_plus_{i}")
            idx_minus = variable_names.index(f"delta_minus_{i}")
            objective_coeffs[idx_plus] += w_early * instance["beta"]
            objective_coeffs[idx_minus] += w_late * instance["gamma"]
            idx_z = variable_names.index(f"z_{i}")
            objective_coeffs[idx_z] -= w_unserved * instance["lambda_demand"][i]

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
                "upper_bounds": variable_bounds_ub,
                "lower_bounds": variable_bounds_lb
            },
            "maximize": False,
            "variable_names": variable_names,
            "variable_types": variable_types,
            "solver_config": {
                "time_limit": 100
            }
        }


    def solve_instance(self, instance):
        data = self.build_problem_data(instance)
        start = time.time()
        solution = self.cuopt_service_client.get_LP_solve(data, incumbent_callback=callback, response_type="dict", logging_callback=log_callback)
        solution = self.repoll(solution)
        elapsed = time.time() - start

        used_vehicles = 0
        objective = None
        if "response" in solution:
            resp = solution["response"]
            #print(json.dumps(resp, indent=4))
            if "solver_response" in resp and "solution" in resp["solver_response"]:
                sol = resp["solver_response"]["solution"]
                vars = sol.get("vars", {})
                objective = sol.get("primal_objective", None)
                used_vehicles = sum(1 for k in range(instance["num_vehicles"]) if vars.get(f"y_{k}", 0) > 0.5)
        return objective, elapsed, used_vehicles

    def batch_solve(self):
        for idx, instance in enumerate(self.instances):
            if idx < 10:
                obj, t, used = self.solve_instance(instance)
                self.stats["objective"].append(obj)
                self.stats["time"].append(t)
                self.stats["used_vehicles"].append(used)
                print(f"Instance {idx+1}/{self.num_instances}: Obj={obj}, Time={t:.2f}s, Used Vehicles={used}")

    def show_stats(self):
        if self.stats["objective"]:
            print(f"\n统计：")
            print(f"平均目标值: {np.mean(self.stats['objective']):.2f}")
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

# 用法示例
if __name__ == "__main__":
    solver = VRPMIPBatchSolver("../data/vrp_dataset_100.json")
    solver.batch_solve()
    solver.show_stats()