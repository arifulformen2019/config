#!/bin/bash

# Define base ports
BASE_PORT=3000
FASTAPI_PORT=8080
OTEL_GRPC_PORT=4317
OTEL_HTTP_PORT=4318
OTEL_PROM_PORT=55679

# Repository URL
REPO_URL="https://github.com/arifulformen2019/rl-swarm.git"

# Temporary directory to clone the original repository
TEMP_DIR="rl-swarm-temp"

# Number of directories to create (can be changed, e.g.: 5 or 10)
NUM_INSTANCES=2

# Port increment step for each instance
INCREMENT_STEP=10

# Check if git and sed are installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed. Please install Git before running this script."
    exit 1
fi
if ! command -v sed &> /dev/null; then
    echo "Error: sed is not installed. Please install sed before running this script."
    exit 1
fi
if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed. Please install curl before running this script."
    exit 1
fi

# Step 1: Clone repository
echo "Cloning repository from $REPO_URL..."
if [ -d "$TEMP_DIR" ]; then
    echo "Directory $TEMP_DIR already exists, removing..."
    rm -rf "$TEMP_DIR"
fi
git clone "$REPO_URL" "$TEMP_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Unable to clone repository."
    exit 1
fi

# Step 2: Create directories from the original clone
for ((i=1; i<=NUM_INSTANCES; i++)); do
    INSTANCE="rl-swarm-$i"
    echo "Creating directory $INSTANCE..."
    if [ -d "$INSTANCE" ]; then
        echo "Directory $INSTANCE already exists, removing..."
        rm -rf "$INSTANCE"
    fi
    cp -r "$TEMP_DIR" "$INSTANCE"
done

# Remove temporary directory
echo "Removing temporary directory $TEMP_DIR..."
rm -rf "$TEMP_DIR"

# Step 3: Change ports for each instance
for ((i=0; i<=NUM_INSTANCES; i++)); do
    INSTANCE="rl-swarm-$i"
    INCREMENT=$(( (i-1) * INCREMENT_STEP ))

    # Calculate new ports
    NEW_BASE_PORT=$((BASE_PORT + INCREMENT))
    NEW_FASTAPI_PORT=$((FASTAPI_PORT + INCREMENT))
    NEW_OTEL_GRPC_PORT=$((OTEL_GRPC_PORT + INCREMENT))
    NEW_OTEL_HTTP_PORT=$((OTEL_HTTP_PORT + INCREMENT))
    NEW_OTEL_PROM_PORT=$((OTEL_PROM_PORT + INCREMENT))

    # File paths
    COMPOSE_FILE="$INSTANCE/docker-compose.yaml"
    RUN_SCRIPT="$INSTANCE/run_rl_swarm.sh"
    PACKAGE_JSON="$INSTANCE/modal-login/package.json"
    ENV_FILE="$INSTANCE/.env"
    CONFIG_YAML="$INSTANCE/rgym_exp/config/rg-swarm.yaml"
    COORDINATOR_PY="$INSTANCE/.venv/lib/python3.12/site-packages/genrl/blockchain/coordinator.py"

    # Update docker-compose.yaml
    if [ -f "$COMPOSE_FILE" ]; then
        echo "Updating ports in $COMPOSE_FILE..."
        sed -i "s/$BASE_PORT/$NEW_BASE_PORT/g" "$COMPOSE_FILE"
        sed -i "s/$FASTAPI_PORT/$NEW_FASTAPI_PORT/g" "$COMPOSE_FILE"
        sed -i "s/$OTEL_GRPC_PORT/$NEW_OTEL_GRPC_PORT/g" "$COMPOSE_FILE"
        sed -i "s/$OTEL_HTTP_PORT/$NEW_OTEL_HTTP_PORT/g" "$COMPOSE_FILE"
        sed -i "s/$OTEL_PROM_PORT/$NEW_OTEL_PROM_PORT/g" "$COMPOSE_FILE"
    else
        echo "Warning: $COMPOSE_FILE not found"
    fi

    # Update run_rl_swarm.sh
    if [ -f "$RUN_SCRIPT" ]; then
        echo "Updating ports in $RUN_SCRIPT..."
        sed -i "s/$BASE_PORT/$NEW_BASE_PORT/g" "$RUN_SCRIPT"
    else
        echo "Warning: $RUN_SCRIPT not found"
    fi

    # Update package.json in modal-login
    if [ -f "$PACKAGE_JSON" ]; then
        echo "Updating ports in $PACKAGE_JSON..."
        sed -i "s/\"next start\"/\"next start --port $NEW_BASE_PORT\"/g" "$PACKAGE_JSON"
    else
        echo "Warning: $PACKAGE_JSON not found"
    fi

    # Update .env (if exists)
    if [ -f "$ENV_FILE" ]; then
        echo "Updating ports in $ENV_FILE..."
        if grep -q "API_PORT=" "$ENV_FILE"; then
            sed -i "s/API_PORT=.*/API_PORT=$NEW_BASE_PORT/g" "$ENV_FILE"
        else
            echo "API_PORT=$NEW_BASE_PORT" >> "$ENV_FILE"
        fi
    else
        echo "Creating new $ENV_FILE with API_PORT..."
        echo "API_PORT=$NEW_BASE_PORT" > "$ENV_FILE"
    fi

    # Update config.yaml (if exists)
    if [ -f "$CONFIG_YAML" ]; then
        echo "Updating ports in $CONFIG_YAML..."
        sed -i "s/port: $BASE_PORT/port: $NEW_BASE_PORT/g" "$CONFIG_YAML"
        sed -i "s/http:\/\/localhost:$BASE_PORT/http:\/\/localhost:$NEW_BASE_PORT/g" "$CONFIG_YAML"
    else
        echo "Warning: $CONFIG_YAML not found"
    fi

    # Update coordinator.py (if needed)
    # if [ -f "$COORDINATOR_PY" ]; then
    #     echo "Checking and updating ports in $COORDINATOR_PY..."
    #     sed -i "s/http:\/\/localhost:$BASE_PORT/http:\/\/localhost:$NEW_BASE_PORT/g" "$COORDINATOR_PY"
    # else
    #     echo "Warning: $COORDINATOR_PY not found"
    # fi

    echo "New ports for $INSTANCE:"
    echo "BASE_PORT: $NEW_BASE_PORT"
    echo "FASTAPI_PORT: $NEW_FASTAPI_PORT"
    echo "OTEL_GRPC_PORT: $NEW_OTEL_GRPC_PORT"
    echo "OTEL_HTTP_PORT: $NEW_OTEL_HTTP_PORT"
    echo "OTEL_PROM_PORT: $NEW_OTEL_PROM_PORT"
done

# Step 4: Check API status
# for ((i=1; i<=NUM_INSTANCES; i++)); do
#     INSTANCE="rl-swarm-$i"
#     NEW_BASE_PORT=$((BASE_PORT + (i-1) * INCREMENT_STEP))
#     echo "Checking API for $INSTANCE on port $NEW_BASE_PORT..."
#     if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$NEW_BASE_PORT/api/submit-reward"; then
#         echo "API /submit-reward on port $NEW_BASE_PORT is ready."
#     else
#         echo "Warning: API /submit-reward on port $NEW_BASE_PORT is not responding. Please run 'yarn start' in $INSTANCE/modal-login first."
#     fi
# done

echo "Complete! $NUM_INSTANCES directories have been created and ports have been updated."
echo "To run each instance:"
echo "1. Go to the modal-login directory and run 'yarn start' or 'npm start'."
echo "2. Then, go to the root directory and run './run_rl_swarm.sh' or 'python rgym_exp/runner/swarm_launcher.py'."
echo "Example:"
echo "  cd rl-swarm-1/modal-login && yarn start"
echo "  cd ../.. && python rgym_exp/runner/swarm_launcher.py"
