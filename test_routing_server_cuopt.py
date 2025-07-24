from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import time

# Example data for routing problem
# The data is structured as per the OpenAPI specification for the server, please refer /cuopt/request -> schema -> OptimizeRoutingData
data = {"cost_matrix_data": {"data": {"0": [[0,1],[1,0]]}},
        "task_data": {"task_locations": [0,1]},
        "fleet_data": {"vehicle_locations": [[0,0],[0,0]]}}

# If cuOpt is not running on localhost:5000, edit ip and port parameters
cuopt_service_client = CuOptServiceSelfHostClient(
    ip="localhost",
    port="5000",
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

solution = cuopt_service_client.get_optimized_routes(data)

# Number of repoll requests to be carried out for a successful response
repoll_tries = 500

solution = repoll(solution, repoll_tries)

print(json.dumps(solution, indent=4))