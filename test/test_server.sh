# Set the server IP and port to be used
SERVER_IP=0.0.0.0
SERVER_PORT=8000

# Start server and store PID
python3 -m cuopt_server.cuopt_service --ip $SERVER_IP --port $SERVER_PORT > cuopt_server.log 2>&1 &
SERVER_PID=$!

# Check if cuOpt server is ready
for i in {1..5}; do
    if [ "$(curl -s -o /dev/null -w "%{http_code}" http://${SERVER_IP}:${SERVER_PORT}/cuopt/health)" = "200" ]; then
        echo "cuOpt server is ready"
        break
    fi
    if [ $i -eq 5 ]; then
        echo "Error: cuOpt server failed to start"
        exit 1
    fi
    sleep 1
done

# Test the server with sample routing problem
# Use /cuopt/request to submit a request to the server
REQID=$(curl --location "http://${SERVER_IP}:${SERVER_PORT}/cuopt/request" \
    --header 'Content-Type: application/json' \
    --header "CLIENT-VERSION: custom" \
    -d '{
        "cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
        "task_data": {"task_locations": [1], "demand": [[1]], "task_time_windows": [[0, 10]], "service_times": [1]},
        "fleet_data": {"vehicle_locations":[[0, 0]], "capacities": [[2]], "vehicle_time_windows":[[0, 20]] },
        "solver_config": {"time_limit": 2}
    }' | jq -r '.reqId')

# Verify we got a 200 response and reqId
if [ -z "$REQID" ]; then
    echo "Error: Failed to get reqId from server"
    exit 1
else
    echo "Successfully submitted request with ID: $REQID"
fi

# Poll for results
# Use /cuopt/solution/${REQID} to poll for results
for i in {1..5}; do
    RESPONSE=$(curl --location "http://${SERVER_IP}:${SERVER_PORT}/cuopt/solution/${REQID}" \
        --header 'Content-Type: application/json' \
        --header "CLIENT-VERSION: custom")

    if echo "$RESPONSE" | jq -e 'has("response")' > /dev/null 2>&1; then
        echo "Got solution response:"
        echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
        break
    else
        echo "Response status:"
        echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
    fi

    if [ $i -eq 5 ]; then
        echo "Error: Timed out waiting for solution"
        exit 1
    fi

    echo "Waiting for solution..."
    sleep 1
done

# Shutdown the server
kill $SERVER_PID