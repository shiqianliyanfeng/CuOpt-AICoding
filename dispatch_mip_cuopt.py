import cuopt
import numpy as np

# 示例数据
num_customers = 4
num_vehicles = 2
depot = 0

# 节点集合（0为配送中心）
nodes = list(range(num_customers + 1))  # [0, 1, 2, 3, 4]
customers = nodes[1:]

# 距离矩阵（对称）
distance_matrix = np.array([
    [0, 10, 20, 15, 25],
    [10, 0, 12, 18, 22],
    [20, 12, 0, 8, 17],
    [15, 18, 8, 0, 13],
    [25, 22, 17, 13, 0]
])

# 客户需求
demands = [0, 3, 2, 4, 1]  # depot为0

# 车辆参数
vehicle_capacities = [5, 7]
vehicle_speeds = [40, 30]  # km/h
vehicle_fixed_costs = [100, 120]
vehicle_cost_per_km = [2, 3]

# 时间窗（e_i, l_i），服务时间
time_windows = [(0, 100), (10, 30), (20, 40), (15, 35), (25, 45)]
service_times = [0, 5, 5, 5, 5]

# 软时间窗惩罚系数
beta = 10  # 早到
gamma = 20 # 迟到

# cuOpt建模
model = cuopt.Model()

# 决策变量
x = {}
for k in range(num_vehicles):
    for i in nodes:
        for j in nodes:
            if i != j:
                x[i, j, k] = model.add_variable(name=f"x_{i}_{j}_{k}", var_type="binary")

y = {}
for k in range(num_vehicles):
    y[k] = model.add_variable(name=f"y_{k}", var_type="binary")

a = {}
for i in nodes:
    a[i] = model.add_variable(name=f"a_{i}", lb=0)

u = {}
for i in nodes:
    for k in range(num_vehicles):
        u[i, k] = model.add_variable(name=f"u_{i}_{k}", lb=0, ub=vehicle_capacities[k])

delta_plus = {}
delta_minus = {}
for i in customers:
    delta_plus[i] = model.add_variable(name=f"delta_plus_{i}", lb=0)
    delta_minus[i] = model.add_variable(name=f"delta_minus_{i}", lb=0)

# 目标函数
fixed_cost = sum(vehicle_fixed_costs[k] * y[k] for k in range(num_vehicles))
transport_cost = sum(
    vehicle_cost_per_km[k] * distance_matrix[i, j] * x[i, j, k]
    for k in range(num_vehicles) for i in nodes for j in nodes if i != j
)
time_window_penalty = sum(
    beta * delta_plus[i] + gamma * delta_minus[i] for i in customers
)
model.set_objective(fixed_cost + transport_cost + time_window_penalty, sense="min")

# 约束1：每个客户仅被服务一次
for i in customers:
    model.add_constraint(
        sum(x[j, i, k] for k in range(num_vehicles) for j in nodes if j != i) == 1
    )

# 约束2：流量守恒（车辆出发和返回）
for k in range(num_vehicles):
    model.add_constraint(
        sum(x[depot, j, k] for j in customers) == y[k]
    )
    model.add_constraint(
        sum(x[j, depot, k] for j in customers) == y[k]
    )

# 约束3：容量约束
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

# 约束4：时间窗约束（软约束）
M = 1e5
for k in range(num_vehicles):
    for i in nodes:
        for j in customers:
            if i != j:
                travel_time = distance_matrix[i, j] / vehicle_speeds[k]
                model.add_constraint(
                    a[j] >= a[i] + service_times[i] + travel_time - M * (1 - x[i, j, k])
                )

for i in customers:
    e_i, l_i = time_windows[i]
    model.add_constraint(delta_plus[i] >= e_i - a[i])
    model.add_constraint(delta_plus[i] >= 0)
    model.add_constraint(delta_minus[i] >= a[i] - l_i)
    model.add_constraint(delta_minus[i] >= 0)

# 约束5：车辆启用关联
for k in range(num_vehicles):
    model.add_constraint(
        sum(x[i, j, k] for i in nodes for j in nodes if i != j) <= M * y[k]
    )

# 求解
solution = model.solve()

# 输出结果
for k in range(num_vehicles):
    print(f"车辆{k}是否启用:", solution[f"y_{k}"])
for i in customers:
    print(f"客户{i}到达时间:", solution[f"a_{i}"], "早到惩罚:", solution[f"delta_plus_{i}"], "迟到惩罚:", solution[f"delta_minus_{i}"])
print("目标值:", solution["objective_value"])