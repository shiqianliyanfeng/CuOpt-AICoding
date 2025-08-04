from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import time

data = {
    "csr_constraint_matrix": {
        "offsets": [0, 2],
        "indices": [0, 1],
        "values": [1.0, 1.0]
    },
    "constraint_bounds": {
        "upper_bounds": [5000.0],
        "lower_bounds": [0.0]
    },
    "objective_data": {
        "coefficients": [1.2, 1.7],
        "scalability_factor": 1.0,
        "offset": 0.0
    },
    "variable_bounds": {
        "upper_bounds": [3000.0, 5000.0],
        "lower_bounds": [0.0, 0.0]
    },
    "maximize": True,
    "variable_names": ["x", "y"],
    "variable_types": ["I", "I"],
    "solver_config":{
        "time_limit": 30
    }
}

# If cuOpt is not running on localhost:5000, edit ip and port parameters
cuopt_service_client = CuOptServiceSelfHostClient(
    ip="0.0.0.0",
    port="5001",
    timeout_exception=False
)

# callback should accept 2 values, one is solution and another is cost
def callback(solution, solution_cost):
    print(f"Solution : {solution} cost : {solution_cost}\n")

# Logging callback
def log_callback(log):
    for i in log:
        print("server-log: ", i)

solution = cuopt_service_client.get_LP_solve(
    data, incumbent_callback=callback, response_type="dict", logging_callback=log_callback
)

print(json.dumps(solution, indent=4))