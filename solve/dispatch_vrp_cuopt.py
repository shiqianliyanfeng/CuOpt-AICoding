from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import time
import numpy as np

# 示例数据
num_customers = 4
num_vehicles = 2
depot = 0

nodes = list(range(num_customers + 1))  # [0, 1, 2, 3, 4]
customers = nodes[1:]

distance_matrix = np.array([
    [0, 10, 20, 15, 25],
    [10, 0, 12, 18, 22],
    [20, 12, 0, 8, 17],
    [15, 18, 8, 0, 13],
    [25, 22, 17, 13, 0]
])

demands = [0, 3, 2, 4, 1]  # depot为0
vehicle_capacities = [5, 7]
vehicle_fixed_costs = [100, 120]
vehicle_speeds = [40, 30]  # km/h
vehicle_cost_per_km = [2, 3]
time_windows = [(0, 100), (10, 30), (20, 40), (15, 35), (25, 45)]
service_times = [0, 5, 5, 5, 5]
early_penalty = 10
late_penalty = 20

# 构造 routing server 所需数据结构
data = {
    "cost_matrix_data": {
        "data": {
            "0": distance_matrix.tolist()
        }
    },
    "task_data": {
        "task_locations": customers,
        "demands": [demands[i] for i in customers],
        "time_windows": [list(tw) for tw in time_windows[1:]],
        "service_times": [service_times[i] for i in customers]
    },
    "fleet_data": {
        "vehicle_locations": [[depot] for _ in range(num_vehicles)],
        "capacities": vehicle_capacities,
        "fixed_costs": vehicle_fixed_costs,
        "speeds": vehicle_speeds,
        "cost_per_distance": vehicle_cost_per_km,
        "time_windows": [list(time_windows[0]) for _ in range(num_vehicles)]
    },
    "penalties": {
        "early": early_penalty,
        "late": late_penalty
    }
}

cuopt_service_client = CuOptServiceSelfHostClient(
    ip="0.0.0.0",
    port=8000,
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

solution = cuopt_service_client.get_optimized_routes(data)
repoll_tries = 500
solution = repoll(solution, repoll_tries)

# 输出结果
print(json.dumps(solution, indent=4))