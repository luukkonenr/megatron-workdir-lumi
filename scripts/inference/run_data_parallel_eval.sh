#!/bin/bash
# Complete workflow for data parallel inference with eval harness

# Configuration
NUM_GPUS=4
BASE_PORT=5000
LAUNCHER="scripts/slurm/srun_inference.sh"  # Adjust to your launcher

echo "========================================="
echo "Data Parallel Inference Workflow"
echo "========================================="
echo ""

# Step 1: Launch servers
echo "Step 1: Launching $NUM_GPUS inference servers..."
bash scripts/inference/run_data_parallel_servers.sh $NUM_GPUS $BASE_PORT $LAUNCHER

# Wait for servers to initialize
echo ""
echo "Waiting 30 seconds for servers to initialize..."
sleep 30

# Step 2: Verify servers are running
echo ""
echo "Step 2: Verifying server health..."
ALL_HEALTHY=true
for i in $(seq 0 $((NUM_GPUS-1))); do
    PORT=$((BASE_PORT + i))
    if curl -s "http://localhost:$PORT/api" > /dev/null 2>&1; then
        echo "✅ GPU $i (port $PORT): OK"
    else
        echo "❌ GPU $i (port $PORT): FAILED"
        ALL_HEALTHY=false
    fi
done

if [ "$ALL_HEALTHY" = false ]; then
    echo ""
    echo "⚠️  Some servers failed to start. Check logs in logs/server_gpu*.log"
    echo "Aborting..."
    exit 1
fi

# Step 3: Run eval harness
echo ""
echo "Step 3: Running eval harness with data parallel inference..."
echo ""

export PYTHONPATH=$PYTHONPATH:lm-evaluation-harness-main

python3 data_parallel_completions_model.py --help || {
    echo "Installing data parallel model..."
    # The model is already created, just need to ensure it's importable
}

python3 lm-evaluation-harness-main/lm_eval/__main__.py \
    --model local-completions \
    --tasks arc_easy \
    --batch_size 64 \
    --model_args model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://localhost:${BASE_PORT}/completions,num_concurrent=1 \
    --limit 100  # Remove this to run full eval

echo ""
echo "========================================="
echo "Eval complete!"
echo "========================================="
echo ""
echo "To stop all servers:"
echo "  bash scripts/inference/stop_data_parallel_servers.sh"
