from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import time

# 构造MILP问题数据
data = {
    "csr_constraint_matrix": {
        "offsets": [0, 2, 4],
        "indices": [0, 1, 0, 1],
        "values": [1.0, 2.0, 3.0, -1.0]
    },
    "constraint_bounds": {
        "upper_bounds": [14.0, float("inf")],
        "lower_bounds": [-float("inf"), 0.0]
    },
    "objective_data": {
        "coefficients": [1.0, 1.0],
        "scalability_factor": 1.0,
        "offset": 0.0
    },
    "variable_bounds": {
        "upper_bounds": [10.0, 10.0],
        "lower_bounds": [0.0, 0.0]
    },
    "maximize": True,
    "variable_names": ["x", "y"],
    "variable_types": ["I", "C"],  # x为整数，y为连续
    "solver_config": {
        "time_limit": 30
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
        for i in range(repoll_tries):
            solution = cuopt_service_client.repoll(req_id, response_type="dict")
            if "reqId" in solution and "response" in solution:
                break
            time.sleep(1)
    return solution

solution = cuopt_service_client.get_LP_solve(data, response_type="dict")
solution = repoll(solution)

# 输出结果
if "response" in solution:
    resp = solution["response"]
    vars = resp["solver_response"]["solution"]["vars"]
    objective = resp["solver_response"]["solution"]["primal_objective"]
    print("x =", vars.get("x"))
    print("y =", vars.get("y"))
    print("目标值 =", objective)
else:
    print(json.dumps(solution, indent=4))