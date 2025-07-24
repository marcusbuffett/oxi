#!/bin/bash

# Example script to run Optuna hyperparameter optimization for Oxi
# Supports both local and remote execution

# Install Python dependencies with uv
echo "Installing Python dependencies..."
uv sync

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Set up data path - adjust this to point to your training data
DATA_PATH="${DATA_PATH:-./tests/data/pgn}"

# Check for remote execution
if [ "$1" = "--remote" ]; then
    echo "=== REMOTE EXECUTION MODE ==="
    if [ -z "$SSH_HOST" ]; then
        echo "Error: SSH_HOST environment variable must be set for remote execution"
        echo "Example: export SSH_HOST=your-server-ip"
        exit 1
    fi
    
    echo "Will run training on ubuntu@$SSH_HOST"
    echo "Make sure:"
    echo "  1. The code is deployed at /home/ubuntu/oxi on the remote server"
    echo "  2. Training data is available at the specified path on the remote server"
    echo "  3. SSH key authentication is set up"
    echo ""
    
    # For remote, don't check local data path
    REMOTE_FLAG="--remote"
    DATA_PATH="${DATA_PATH:-/home/ubuntu/data}"  # Default remote path
else
    echo "=== LOCAL EXECUTION MODE ==="
    # Check if data path exists locally
    if [ ! -d "$DATA_PATH" ]; then
        echo "Error: Data path $DATA_PATH does not exist"
        echo "Please set DATA_PATH environment variable or ensure test data exists"
        echo "For remote execution, use: $0 --remote"
        exit 1
    fi
    REMOTE_FLAG=""
fi

# Run optimization
echo "Starting Optuna optimization..."
echo "Data path: $DATA_PATH"
echo "This will run 50 trials with progressive pruning..."

uv run python optimize.py \
    --data-path "$DATA_PATH" \
    --n-trials 10 \
    --study-name "oxi-hyperopt-$(date +%Y%m%d-%H%M%S)" \
    $REMOTE_FLAG

echo "Optimization complete!"
