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
vehicle_fixed_costs = [100, 120]
vehicle_speeds = [40, 30]  # km/h
vehicle_cost_per_km = [2, 3]

# 时间窗（e_i, l_i），服务时间
time_windows = [(0, 100), (10, 30), (20, 40), (15, 35), (25, 45)]
service_times = [0, 5, 5, 5, 5]

# 软时间窗惩罚系数
early_penalty = 10
late_penalty = 20

# cuOpt VRP建模
solver = cuopt.Solver()

# 设置距离矩阵
solver.set_distance_matrix(distance_matrix)

# 设置车辆参数
solver.set_vehicle_capacity(vehicle_capacities)
solver.set_vehicle_fixed_cost(vehicle_fixed_costs)
solver.set_vehicle_speed(vehicle_speeds)
solver.set_vehicle_cost_per_distance(vehicle_cost_per_km)

# 设置客户需求
solver.set_demands(demands)

# 设置时间窗和服务时间
solver.set_time_windows([tw for tw in time_windows])
solver.set_service_times(service_times)

# 设置软时间窗惩罚
solver.set_time_window_penalty(early_penalty, late_penalty)

# 求解
result = solver.solve()

# 输出结果
print("目标值:", result.objective)
for k, route in enumerate(result.routes):
    print(f"车辆{k}路径:", route)
    print(f"车辆{k}是否启用:", int(len(route) > 2))
for i in customers:
    print(f"客户{i}到达时间:", result.arrival_times[i])
    print(f"客户{i}早到惩罚:", result.early_penalties[i])
    print(f"客户{i}迟到惩罚:", result.late_penalties[i])