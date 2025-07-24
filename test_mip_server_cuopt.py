from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import time

# Example data for MILP problem
# The data is structured as per the OpenAPI specification for the server, please refer /cuopt/request -> schema -> LPData
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
    port=8000,
    polling_timeout=25,
    timeout_exception=False
)

def repoll(solution, repoll_tries):
    # If solver is still busy solving, the job will be assigned a request id and response is sent back in the
    # following format {"reqId": <REQUEST-ID>}.
    # Solver needs to be re-polled for response using this <REQUEST-ID>.

    if "reqId" in solution and "response" not in solution:
        req_id = solution["reqId"]
        for i in range(repoll_tries):
            solution = cuopt_service_client.repoll(req_id, response_type="dict")
            if "reqId" in solution and "response" in solution:
                break;

            # Sleep for a second before requesting
            time.sleep(1)

    return solution

solution = cuopt_service_client.get_LP_solve(data, response_type="dict")

# Number of repoll requests to be carried out for a successful response
repoll_tries = 500

solution = repoll(solution, repoll_tries)

print(json.dumps(solution, indent=4))