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
    demands = [0] + [random.randint(1, 4) for _ in customers]
    vehicle_capacities = [random.randint(15, 30) for _ in range(num_vehicles)]
    vehicle_fixed_costs = [random.randint(0, 10) for _ in range(num_vehicles)]
    vehicle_cost_per_km = [round(random.uniform(0.005, 0.015), 2) for _ in range(num_vehicles)]
    cost_matrix = []
    for k in range(num_vehicles):
        cost_matrix.append([[round(distance_matrix[i][j] * vehicle_cost_per_km[k], 2) for j in nodes] for i in nodes])

    time_windows = [(0, 100)]
    for _ in customers:
        e = random.randint(0, 0)
        l = e + random.randint(70, 90)
        time_windows.append((e, l))
    service_times = [0] + [random.randint(3, 10) for _ in customers]
    
    early_penalty = 10
    late_penalty = 20
    # 权重参数
    w_fixed = 1.0
    w_distance = 1.0
    w_early = 1.0
    w_late = 1.0
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
        "vehicle_fixed_costs": vehicle_fixed_costs,
        "vehicle_cost_per_km": vehicle_cost_per_km,
        "cost_matrix": cost_matrix,
        "time_windows": time_windows,
        "service_times": service_times,
        "early_penalty": early_penalty,
        "late_penalty": late_penalty,
        "w_fixed": w_fixed,
        "w_distance": w_distance,
        "w_early": w_early,
        "w_late": w_late
    }

model_data = gen_model_data()
print(model_data)

num_vehicles = model_data["num_vehicles"]
depot = model_data["depot"]
customers = model_data["customers"]
distance_matrix = model_data["distance_matrix"]
demands = model_data["demands"]
vehicle_capacities = model_data["vehicle_capacities"]
vehicle_fixed_costs = model_data["vehicle_fixed_costs"]
vehicle_cost_per_km = model_data["vehicle_cost_per_km"]
cost_matrix = model_data["cost_matrix"]
time_windows = model_data["time_windows"]
service_times = model_data["service_times"]
early_penalty = model_data["early_penalty"]
late_penalty = model_data["late_penalty"]
w_fixed = model_data["w_fixed"]
w_distance = model_data["w_distance"]
w_early = model_data["w_early"]
w_late = model_data["w_late"]

# 构造 routing server 所需数据结构
data = {
    "cost_matrix_data": {
        "data": {
            "0": distance_matrix
        }
    },
    "task_data": {
        "task_locations": customers,
        "demand": [[demands[i] for i in customers]],
        #"task_time_windows": [list(time_windows[i]) for i in customers],
        #"service_times": [service_times[i] for i in customers]
    },
    "fleet_data": {
        "vehicle_locations": [[0,0] for _ in range(num_vehicles)],
        "capacities": [vehicle_capacities]
        #"vehicle_fixed_costs": [w_fixed * fc for fc in vehicle_fixed_costs],
    },
    "solver_config": {"time_limit": 20}
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

solution = cuopt_service_client.get_optimized_routes(data)
solution = repoll(solution)

print(json.dumps(solution, indent=4))
# 输出结果
if "response" in solution:
    resp = solution["response"]
    print("目标值:", resp.get("objective_value"))
    for idx, route in enumerate(resp.get("routes", [])):
        print(f"车辆{idx}路径:", route)
    if "arrival_times" in resp:
        print("到达时间:", resp["arrival_times"])
    if "early_penalties" in resp:
        print("早到惩罚:", resp["early_penalties"])
    if "late_penalties" in resp:
        print("迟到惩罚:", resp["late_penalties"])
else:
    print(json.dumps(solution, indent=4))