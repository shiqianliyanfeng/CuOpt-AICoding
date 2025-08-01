from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import time
import numpy as np
import random

def gen_model_data():
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
    lambda_demand = [0] + [round(500 * demands[i], 2) for i in customers]
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
        "lambda_demand": lambda_demand,  # 示例，10为η
        "eta": eta,
        "cost_matrix": cost_matrix,
        "w_fixed": 1.0,
        "w_distance": 1.0,
        "w_early": 1.0,
        "w_late": 1.0,
        "w_unserved": 1.0,
    }

model_data = gen_model_data()
print(model_data)

# 变量名和类型
variable_names = []
variable_types = []
variable_bounds_lb = []
variable_bounds_ub = []

# x[i,j,k]: 是否车辆k从i到j
x_indices = []
for k in range(model_data["num_vehicles"]):
    for i in model_data["nodes"]:
        for j in model_data["nodes"]:
            if i != j:
                variable_names.append(f"x_{i}_{j}_{k}")
                variable_types.append("B")
                variable_bounds_lb.append(0)
                variable_bounds_ub.append(1)
                x_indices.append((i, j, k))

# y[k]: 车辆k是否启用
y_indices = []
for k in range(model_data["num_vehicles"]):
    variable_names.append(f"y_{k}")
    variable_types.append("B")
    variable_bounds_lb.append(0)
    variable_bounds_ub.append(1)
    y_indices.append(k)

# a[i]: 到达时间
a_indices = []
for i in model_data["nodes"]:
    variable_names.append(f"a_{i}")
    variable_types.append("C")
    variable_bounds_lb.append(0)
    variable_bounds_ub.append(1e5)
    a_indices.append(i)

# u[i,k]: 车辆k在节点i的载重
u_indices = []
for i in model_data["nodes"]:
    for k in range(model_data["num_vehicles"]):
        variable_names.append(f"u_{i}_{k}")
        variable_types.append("C")
        variable_bounds_lb.append(0)
        variable_bounds_ub.append(model_data["vehicle_capacities"][k])
        u_indices.append((i, k))

# delta_plus_indices, delta_minus_indices: 软时间窗惩罚
delta_plus_indices = []
delta_minus_indices = []
for i in model_data["customers"]:
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

# z[i]: 客户i是否被服务
z_indices = []
for i in model_data["customers"]:
    variable_names.append(f"z_{i}")
    variable_types.append("B")
    variable_bounds_lb.append(0)
    variable_bounds_ub.append(1)
    z_indices.append(i)

# 约束和目标函数转CSR格式（此处仅举例，实际需根据你的模型完整展开）
# 这里只写部分约束和目标，完整模型需补充所有约束
csr_offsets = [0]
csr_indices = []
csr_values = []
upper_bounds = []
lower_bounds = []

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

# 约束：每个客户被访问次数小于等于z_i
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
                lower_bounds.append(-np.inf)

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

M = 1e5
# 约束：车辆容量变化
for k in range(model_data["num_vehicles"]):
    for i in model_data["nodes"]:
        for j in model_data["customers"]:
            if j != i:
                row_indices = []
                row_values = []

                idx_u_i = variable_names.index(f"u_{i}_{k}")
                row_indices.append(idx_u_i)
                row_values.append(1.0)

                idx_u_j = variable_names.index(f"u_{j}_{k}")
                row_indices.append(idx_u_j)
                row_values.append(-1.0)

                idx_z_j = variable_names.index(f"z_{j}")
                row_indices.append(idx_z_j)
                row_values.append(model_data["demands"][i])

                idx_x_ijk = variable_names.index(f"x_{j}_{i}_{k}")
                row_indices.append(idx_x_ijk)
                row_values.append(model_data["vehicle_capacities"][k])

                csr_indices.extend(row_indices)
                csr_values.extend(row_values)
                csr_offsets.append(len(csr_indices))
                upper_bounds.append(model_data["vehicle_capacities"][k])
                lower_bounds.append(0.0)

for k in range(model_data["num_vehicles"]):
    for i in model_data["customers"]:
                row_indices = []
                row_values = []

                idx_u_i = variable_names.index(f"u_{i}_{k}")
                row_indices.append(idx_u_i)
                row_values.append(-1.0)

                idx_z_i = variable_names.index(f"z_{i}")
                row_indices.append(idx_z_i)
                row_values.append(model_data["demands"][i])

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
    lower_bounds.append(-model_data["time_windows"][i][0])

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

# 约束：每辆车初始载重为0
for k in range(model_data["num_vehicles"]):
    row_indices = []
    row_values = []
    idx_u_0 = variable_names.index(f"u_{0}_{k}")
    row_indices.append(idx_u_0)
    row_values.append(1.0)
    csr_indices.extend(row_indices)
    csr_values.extend(row_values)
    csr_offsets.append(len(csr_indices))
    upper_bounds.append(0.0)
    lower_bounds.append(0.0)

idx_a_0 = variable_names.index(f"a_{0}")
row_indices = [idx_a_0]
row_values = [1.0]
csr_indices.extend(row_indices)
csr_values.extend(row_values)
csr_offsets.append(len(csr_indices))
upper_bounds.append(0.0)
lower_bounds.append(0.0)

for k in range(model_data["num_vehicles"]):
    row_indices = []
    row_values = []
    for i in model_data["customers"]:
        for j in model_data["customers"]:
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

for idx, k in enumerate(y_indices):
    objective_coeffs[variable_names.index(f"y_{k}")] += w_fixed * model_data["vehicle_fixed_costs"][k]
for (i, j, k) in x_indices:
    idx = variable_names.index(f"x_{i}_{j}_{k}")
    objective_coeffs[idx] += w_distance * model_data["vehicle_cost_per_km"][k] * model_data["distance_matrix"][i][j]
for i in model_data["customers"]:
    idx_plus = variable_names.index(f"delta_plus_{i}")
    idx_minus = variable_names.index(f"delta_minus_{i}")
    objective_coeffs[idx_plus] += w_early * model_data["beta"]
    objective_coeffs[idx_minus] += w_late * model_data["gamma"]
    idx_z = variable_names.index(f"z_{i}")
    objective_coeffs[idx_z] -= w_unserved * model_data["lambda_demand"][i]

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
        "time_limit": 300
    }
}

cuopt_service_client = CuOptServiceSelfHostClient(
    ip="0.0.0.0",
    port="5001",
    polling_timeout=25,
    timeout_exception=False
)

def repoll(solution, repoll_tries=500):
    if "reqId" in solution and "response" not in solution:
        req_id = solution["reqId"]
        for _ in range(repoll_tries):
            solution = cuopt_service_client.repoll(req_id, response_type="dict")
            if "reqId" in solution and "response" in solution:
                break
            time.sleep(1)
    return solution

solution = cuopt_service_client.get_LP_solve(data, response_type="dict")
solution = repoll(solution)

if "response" in solution:
    resp = solution["response"]
    if "solver_response" in resp and "solution" in resp["solver_response"]:
        sol = resp["solver_response"]["solution"]
        vars = sol.get("vars", {})
        objective = sol.get("primal_objective", None)
        for k in y_indices:
            print(f"车辆{k}是否启用:", vars.get(f"y_{k}"))
        for i in a_indices:
            print(f"客户{i}到达时间:", vars.get(f"a_{i}"),
                "早到惩罚:", vars.get(f"delta_plus_{i}"),
                "迟到惩罚:", vars.get(f"delta_minus_{i}"))
        print("目标值:", objective)
    else:
        print(json.dumps(resp, indent=4))
else:
    print(json.dumps(solution, indent=4))
        