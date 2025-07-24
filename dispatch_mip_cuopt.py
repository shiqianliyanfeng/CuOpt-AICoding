from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import time
import numpy as np

# 示例数据（与原始一致）
num_customers = 4
num_vehicles = 2
depot = 0
nodes = list(range(num_customers + 1))
customers = nodes[1:]
distance_matrix = np.array([
    [0, 10, 20, 15, 25],
    [10, 0, 12, 18, 22],
    [20, 12, 0, 8, 17],
    [15, 18, 8, 0, 13],
    [25, 22, 17, 13, 0]
])
demands = [0, 3, 2, 4, 1]
vehicle_capacities = [5, 7]
vehicle_speeds = [40, 30]
vehicle_fixed_costs = [100, 120]
vehicle_cost_per_km = [2, 3]
time_windows = [(0, 100), (10, 30), (20, 40), (15, 35), (25, 45)]
service_times = [0, 5, 5, 5, 5]
beta = 10
gamma = 20

# 变量名和类型
variable_names = []
variable_types = []
variable_bounds_lb = []
variable_bounds_ub = []

# x[i,j,k]: 是否车辆k从i到j
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

# y[k]: 车辆k是否启用
y_indices = []
for k in range(num_vehicles):
    variable_names.append(f"y_{k}")
    variable_types.append("B")
    variable_bounds_lb.append(0)
    variable_bounds_ub.append(1)
    y_indices.append(k)

# a[i]: 到达时间
a_indices = []
for i in nodes:
    variable_names.append(f"a_{i}")
    variable_types.append("C")
    variable_bounds_lb.append(0)
    variable_bounds_ub.append(1e5)
    a_indices.append(i)

# u[i,k]: 车辆k在节点i的载重
u_indices = []
for i in nodes:
    for k in range(num_vehicles):
        variable_names.append(f"u_{i}_{k}")
        variable_types.append("C")
        variable_bounds_lb.append(0)
        variable_bounds_ub.append(vehicle_capacities[k])
        u_indices.append((i, k))

# delta_plus[i], delta_minus[i]: 软时间窗惩罚
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

# 约束和目标函数转CSR格式（此处仅举例，实际需根据你的模型完整展开）
# 这里只写部分约束和目标，完整模型需补充所有约束
csr_offsets = [0]
csr_indices = []
csr_values = []
upper_bounds = []
lower_bounds = []

# 约束1：每个客户仅被服务一次
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
for idx, k in enumerate(y_indices):
    objective_coeffs[variable_names.index(f"y_{k}")] += vehicle_fixed_costs[k]
for (i, j, k) in x_indices:
    idx = variable_names.index(f"x_{i}_{j}_{k}")
    objective_coeffs[idx] += vehicle_cost_per_km[k] * distance_matrix[i, j]
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

cuopt_service_client = CuOptServiceSelfHostClient(
    ip="localhost",
    port=5000,
    polling_timeout=25,
    timeout_exception=False
)

def repoll(solution, repoll_tries):
    if "reqId" in solution and "response" not in solution:
        req_id = solution["reqId"]
        for i in range(repoll_tries):
            solution = cuopt_service_client.repoll(req_id, response_type="dict")
            if "reqId" in solution and "response" in solution:
                break
            time.sleep(1)
    return solution

solution = cuopt_service_client.get_LP_solve(data, response_type="dict")
repoll_tries = 500
solution = repoll(solution, repoll_tries)

# 输出结果
if "response" in solution:
    resp = solution["response"]
    for k in y_indices:
        print(f"车辆{k}是否启用:", resp["variable_values"][variable_names.index(f"y_{k}")])
    for i in customers:
        print(f"客户{i}到达时间:", resp["variable_values"][variable_names.index(f"a_{i}")],
              "早到惩罚:", resp["variable_values"][variable_names.index(f"delta_plus_{i}")],
              "迟到惩罚:", resp["variable_values"][variable_names.index(f"delta_minus_{i}")])
    print("目标值:", resp["objective_value"])
else:
    print(json.dumps(solution, indent=4))