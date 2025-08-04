import numpy as np
import json
import random

def gen_one_instance(num_customers=10, num_vehicles=3):
    depot = 0
    nodes = list(range(num_customers + 1))  # 0为depot
    customers = nodes[1:]

    # 坐标和距离矩阵
    coords = np.random.rand(num_customers + 1, 2) * 100
    distance_matrix = np.round(np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2), 2).tolist()

    # 客户需求
    demands = [0] + [random.randint(1, 5) for _ in customers]

    # 车辆参数
    vehicle_capacities = [random.randint(15, 30) for _ in range(num_vehicles)]
    vehicle_speeds = [random.randint(30, 60) for _ in range(num_vehicles)]
    vehicle_fixed_costs = [random.randint(0, 10) for _ in range(num_vehicles)]
    vehicle_cost_per_km = [round(random.uniform(0.05, 0.15), 2) for _ in range(num_vehicles)]

    # 时间窗和服务时间
    time_windows = [(0, 100)]  # depot
    for _ in customers:
        e = random.randint(10, 30)
        l = e + random.randint(10, 30)
        time_windows.append((e, l))
    service_times = [0] + [random.randint(3, 10) for _ in customers]

    # 软时间窗惩罚系数
    beta = 10
    gamma = 20

    # 未满足需求的惩罚系数
    eta = 10
    lambda_demand = [0] + [round(50 * demands[i], 2) for i in customers]

    # 单位距离成本矩阵 c_{ij}^k * d_{ij}
    cost_matrix = []
    for k in range(num_vehicles):
        cost_matrix.append([[round(distance_matrix[i][j] * vehicle_cost_per_km[k], 2) for j in nodes] for i in nodes])

    # 权重参数
    w_fixed = 1.0
    w_distance = 1.0
    w_early = 1.0
    w_late = 1.0
    w_unserved = 1.0

    return {
        "num_customers": num_customers,
        "num_vehicles": num_vehicles,
        "depot": depot,
        "nodes": nodes,
        "customers": customers,
        "coords": coords.tolist(),
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
        "eta": eta,
        "lambda_demand": lambda_demand,
        "cost_matrix": cost_matrix,
        "w_fixed": w_fixed,
        "w_distance": w_distance,
        "w_early": w_early,
        "w_late": w_late,
        "w_unserved": w_unserved
    }

# 生成100组数据
dataset = []
for _ in range(100):
    num_customers = random.randint(8, 15)
    num_vehicles = random.randint(2, 5)
    instance = gen_one_instance(num_customers, num_vehicles)
    dataset.append(instance)

with open("./data/vrp_dataset_100.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("已生成并保存100组VRP数据集到 vrp_dataset_100.json")