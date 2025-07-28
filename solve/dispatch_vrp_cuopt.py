from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import time
import numpy as np

def gen_model_data():
    num_customers = 4
    num_vehicles = 2
    depot = 0
    nodes = list(range(num_customers + 1))
    customers = nodes[1:]
    coords = np.random.rand(num_customers + 1, 2) * 100
    distance_matrix = np.round(np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2), 2).tolist()
    demands = [0] + [3, 2, 4, 1]
    vehicle_capacities = [[5, 7]]
    vehicle_fixed_costs = [100, 120]
    vehicle_speeds = [40, 30]
    vehicle_cost_per_km = [2, 3]
    time_windows = [(0, 100), (10, 30), (20, 40), (15, 35), (25, 45)]
    service_times = [0, 5, 5, 5, 5]
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
        "vehicle_speeds": vehicle_speeds,
        "vehicle_cost_per_km": vehicle_cost_per_km,
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
num_vehicles = model_data["num_vehicles"]
depot = model_data["depot"]
customers = model_data["customers"]
distance_matrix = np.array(model_data["distance_matrix"])
demands = model_data["demands"]
vehicle_capacities = model_data["vehicle_capacities"]
vehicle_fixed_costs = model_data["vehicle_fixed_costs"]
vehicle_speeds = model_data["vehicle_speeds"]
vehicle_cost_per_km = model_data["vehicle_cost_per_km"]
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
            "0": distance_matrix.tolist()
        }
    },
    "task_data": {
        "task_locations": customers,
        "demand": [[demands[i] for i in customers]],
        "task_time_windows": [list(time_windows[i]) for i in customers],
        "service_times": [service_times[i] for i in customers]
    },
    "fleet_data": {
        "capacities": vehicle_capacities,
        "vehicle_fixed_costs": [w_fixed * fc for fc in vehicle_fixed_costs]
    }
}

cuopt_service_client = CuOptServiceSelfHostClient(
    ip="0.0.0.0",
    port="5000",
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