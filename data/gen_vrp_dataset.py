import numpy as np
import json
import random

def gen_one_instance(num_customers=10, num_vehicles=3):
    depot = 0
    nodes = list(range(num_customers + 1))  # 0为depot
    customers = nodes[1:]

    # 坐标和距离矩阵
    coords = np.random.rand(num_customers + 1, 2) * 100
    distance_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    distance_matrix = np.round(distance_matrix, 2).tolist()

    # 客户需求
    demands = [0] + [random.randint(1, 10) for _ in customers]

    # 未服务惩罚系数 η 和 λ_i
    eta = round(random.uniform(10, 20), 2)
    lambda_demand = [0] + [round(eta * demands[i], 2) for i in customers]

    # 车辆参数
    vehicle_capacities = [random.randint(15, 25) for _ in range(num_vehicles)]      # Q_k
    vehicle_speeds = [random.randint(30, 60) for _ in range(num_vehicles)]          # v_k
    vehicle_fixed_costs = [random.randint(80, 150) for _ in range(num_vehicles)]    # F_k
    vehicle_cost_per_km = [round(random.uniform(1.5, 3.5), 2) for _ in range(num_vehicles)] # c_{ij}^k

    # 时间窗和服务时间
    time_windows = [(0, 100)]  # depot
    for _ in customers:
        e = random.randint(10, 40)
        l = e + random.randint(20, 40)
        time_windows.append((e, l))
    service_times = [0] + [random.randint(3, 10) for _ in customers]

    # 单位距离成本矩阵 c_{ij}^k * d_{ij}
    cost_matrix = []
    for k in range(num_vehicles):
        cost_matrix.append([[round(distance_matrix[i][j] * vehicle_cost_per_km[k], 2) for j in nodes] for i in nodes])

    # 软时间窗惩罚系数
    beta = random.randint(8, 15)
    gamma = random.randint(15, 25)

    return {
        "num_customers": num_customers,
        "num_vehicles": num_vehicles,
        "depot": depot,
        "nodes": nodes,
        "customers": customers,
        "coords": coords.tolist(),
        "distance_matrix": distance_matrix,
        "demands": demands,
        "lambda_demand": lambda_demand,
        "eta": eta,
        "vehicle_capacities": vehicle_capacities,
        "vehicle_speeds": vehicle_speeds,
        "vehicle_fixed_costs": vehicle_fixed_costs,
        "vehicle_cost_per_km": vehicle_cost_per_km,
        "cost_matrix": cost_matrix,
        "time_windows": time_windows,
        "service_times": service_times,
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

with open("vrp_dataset_100.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("已生成并保存100组VRP数据集到 vrp_dataset_100.json")