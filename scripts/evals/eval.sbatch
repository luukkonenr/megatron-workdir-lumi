#!/bin/bash
#SBATCH --job-name=eval-llama-8b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

set -euo pipefail

# ============================================
# CONFIGURATION - Edit these values
# ============================================
CHECKPOINT="/shared_silo/scratch/rluukkon/Megatron-Bridge/checkpoints/llama31-8b-bridge-test"
# TASKS="arc_easy,arc_challenge"
# TASKS="arc_easy,arc_challenge"
TASKS="hellaswag"

# Optimized settings for speed
BATCH_SIZE=256               # Increased from 64 for better throughput
INFERENCE_MAX_BATCH=256      # Match eval batch size
NUM_CONCURRENT=4             # Allow request pipelining
# For dev purposes
num_samples=2000

PORT=5000
HF_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
SEQ_LENGTH=4096
# loads $MODEL_ARGS
source configs/llama3.1-8B.sh 
MODEL_ARGS+=(
    --load $CHECKPOINT
)


# ============================================
# Setup
# ============================================
source environment.sh
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONPATH=$PYTHONPATH:lm-evaluation-harness-main

mkdir -p logs results
overlay="/shared_silo/scratch/rluukkon/overlays/meg_eval.img:ro"
# Cleanup on exit
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT SIGINT SIGTERM

        # --tokenizer-model meta-llama/Llama-3.1-8B-Instruct \
        # --tokenizer-type HuggingFaceTokenizer \
        # --load $CHECKPOINT \
        # --use-checkpoint-args \
        # --max-position-embeddings 131072 \
        # --seq-length $SEQ_LENGTH \
# ============================================
# Launch Server
# ============================================
# 
TEXT_GENERATION_SERVER_ARGS=(
    --use-mcore-models
    --micro-batch-size 8
    --no-load-optim --no-load-rng
    --inference-max-batch-size $INFERENCE_MAX_BATCH
    --inference-max-requests $INFERENCE_MAX_BATCH
    --inference-max-seq-length $SEQ_LENGTH
    --inference-dynamic-batching-buffer-size-gb 150.0
    --inference-dynamic-batching
    --port $PORT
    "${MODEL_ARGS[@]}"
)
# echo "TEXT_GENERATION_SERVER_ARGS: ${TEXT_GENERATION_SERVER_ARGS[@]}"
# echo "Starting inference server..."
# srun --label apptainer exec --overlay $overlay $CONTAINER bash -c "
#     export RANK=\$SLURM_PROCID; export LOCAL_RANK=\$SLURM_LOCALID;
#     torchrun --nproc_per_node=\$SLURM_GPUS_PER_NODE \
#         --nnodes=\$SLURM_JOB_NUM_NODES --node_rank=\$SLURM_NODEID \
#         --rdzv_id=\$SLURM_JOB_ID --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#         Megatron-LM/tools/run_text_generation_server.py \
#         $(printf "%s " "${TEXT_GENERATION_SERVER_ARGS[@]}")
#         " &> logs/server-$SLURM_JOBID.log &
# SERVER_PID=$!

# # Wait for server ready
# echo "Waiting for server..."
# for i in {1..60}; do
#     sleep 2
#     curl -s http://localhost:$PORT/api > /dev/null 2>&1 && break
# done
# echo "Server ready"

# # ============================================
# # Run Evaluations
# # ============================================
# echo "Running evaluations..."
# IFS=',' read -ra TASK_ARRAY <<< "$TASKS"

# for task in "${TASK_ARRAY[@]}"; do
#     echo "Task: $task"
#     # apptainer exec --overlay $overlay $CONTAINER bash -c "
#     srun --label --overlap apptainer exec --overlay $overlay $CONTAINER bash -c "
#     python3 lm-evaluation-harness-main/lm_eval/__main__.py \
#         --model local-completions \
#         --tasks "$task" \
#         --batch_size $BATCH_SIZE \
#         --limit $num_samples \
#         --model_args "model=$HF_MODEL_NAME,base_url=http://localhost:${PORT}/completions,num_concurrent=$NUM_CONCURRENT,max_retries=3,tokenized_requests=True" \
#         --output_path "results/${task}_${SLURM_JOBID}.json"
#     "
# done

# echo "Complete! Results in results/"

# kill $SERVER_PID

HF_MODEL_NAME="/shared_silo/scratch/rluukkon/oellm/megatron2huggingface/llama3.1-8b_hf"
# Compare with hf model
echo "Comparing with hf model..."
apptainer exec --overlay $overlay $CONTAINER bash -c "
    python3 lm-evaluation-harness-main/lm_eval/__main__.py \
        --model hf \
        --limit $num_samples \
        --tasks $TASKS \
        --batch_size $BATCH_SIZE \
        --model_args "pretrained=$HF_MODEL_NAME,trust_remote_code=True" \
        --output_path "results/hf_${SLURM_JOBID}.json"
"