#!/bin/bash
# Launch multiple inference server instances for data parallel inference
# Usage: bash run_data_parallel_servers.sh <num_gpus> <base_port> <launcher_script>

NUM_GPUS=${1:-4}
BASE_PORT=${2:-5000}
LAUNCHER=${3:-"scripts/slurm/srun_inference.sh"}

echo "========================================="
echo "Data Parallel Inference Server Launcher"
echo "========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Base Port: $BASE_PORT"
echo "Launcher: $LAUNCHER"
echo ""

# Clean up any existing PID file
rm -f /tmp/inference_servers.pids

# Launch servers on each GPU
for i in $(seq 0 $((NUM_GPUS-1))); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$i
    
    echo "[GPU $GPU_ID] Starting server on port $PORT..."
    
    # Launch server in background with specific GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID bash scripts/inference/run_text_generation.sh $LAUNCHER --port $PORT > logs/server_gpu${GPU_ID}_port${PORT}.log 2>&1 &
    
    # Store PID
    PID=$!
    echo $PID >> /tmp/inference_servers.pids
    echo "[GPU $GPU_ID] PID: $PID"
    
    # Give it a moment to start
    sleep 2
done

echo ""
echo "✅ All $NUM_GPUS servers launched!"
echo ""
echo "Server URLs:"
for i in $(seq 0 $((NUM_GPUS-1))); do
    PORT=$((BASE_PORT + i))
    echo "  GPU $i: http://localhost:$PORT/completions"
done

echo ""
echo "PIDs stored in: /tmp/inference_servers.pids"
echo "Logs in: logs/server_gpu*_port*.log"
echo ""
echo "To stop all servers:"
echo "  kill \$(cat /tmp/inference_servers.pids)"
echo ""
echo "To check server health:"
echo "  curl http://localhost:$BASE_PORT/api"
