import numpy as np
import json
import random

def gen_one_instance(num_customers=10, num_vehicles=3):
    # 节点集合
    nodes = list(range(num_customers + 1))  # 0为depot
    customers = nodes[1:]

    # 距离矩阵（对称）
    coords = np.random.rand(num_customers + 1, 2) * 100
    distance_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    distance_matrix = np.round(distance_matrix, 2).tolist()

    # 客户需求
    demands = [0] + [random.randint(1, 10) for _ in customers]

    # 车辆参数
    vehicle_capacities = [random.randint(15, 25) for _ in range(num_vehicles)]
    vehicle_speeds = [random.randint(30, 60) for _ in range(num_vehicles)]
    vehicle_fixed_costs = [random.randint(80, 150) for _ in range(num_vehicles)]
    vehicle_cost_per_km = [round(random.uniform(1.5, 3.5), 2) for _ in range(num_vehicles)]

    # 时间窗和服务时间
    time_windows = [(0, 100)]  # depot
    for _ in customers:
        e = random.randint(10, 40)
        l = e + random.randint(20, 40)
        time_windows.append((e, l))
    service_times = [0] + [random.randint(3, 10) for _ in customers]

    # 单位距离成本矩阵
    cost_matrix = []
    for k in range(num_vehicles):
        cost_matrix.append([[round(distance_matrix[i][j] * vehicle_cost_per_km[k], 2) for j in nodes] for i in nodes])

    # 软时间窗惩罚系数
    beta = random.randint(8, 15)
    gamma = random.randint(15, 25)

    return {
        "num_customers": num_customers,
        "num_vehicles": num_vehicles,
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
        "cost_matrix": cost_matrix,
        "beta": beta,
        "gamma": gamma
    }

# 生成100组数据
dataset = []
for _ in range(100):
    num_customers = random.randint(8, 15)
    num_vehicles = random.randint(2, 5)
    instance = gen_one_instance(num_customers, num_vehicles)
    dataset.append(instance)

# 保存为JSON文件
with open("vrp_dataset_100.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("已生成并保存100组VRP数据集到 vrp_dataset_100.json")