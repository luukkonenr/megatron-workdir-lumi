#!/bin/bash

source configs/llama3.1-8B.sh
MODEL_ARGS+=(
    --load /shared_silo/scratch/rluukkon/Megatron-Bridge/checkpoints/llama31-8b-bridge-test
)

timestamp=$(date +%s)
OUTPUT_DIR="eval_results/"
OUTPUT_FILE="${OUTPUT_DIR}/run_${timestamp}"
mkdir -p $OUTPUT_DIR
echo Final results will be saved to: $OUTPUT_FILE

# Adding lm-evaluation-harness to PYTHONPATH without installing it for dev purposes
export PYTHONPATH=$PYTHONPATH:lm-evaluation-harness
export PYTHONPATH=$PYTHONPATH:Megatron-LM

# Optimized inference parameters
# - inference-max-batch-size: Should be >= batch_size to avoid KV cache reallocation
#   For loglikelihood tasks, we only need prompt processing, so can use larger batches
# - inference-max-seq-length: Set based on actual task requirements (most tasks < 2048)
#   Lower values = less KV cache memory = can use larger batches
# - inference-batch-times-seqlen-threshold: Controls attention kernel selection
BATCH_SIZE=64  # Increased from 16 - test and increase further if memory allows
INFERENCE_MAX_BATCH_SIZE=$((BATCH_SIZE * 2))  # Set higher than batch_size for safety
INFERENCE_MAX_SEQ_LENGTH=2048  # Most eval tasks don't need full 8192, saves memory

megatron_arguments=(
        --no-load-optim 
        --no-load-rng 
        --max-tokens-to-oom 40000
        --micro-batch-size 1
        --inference-max-batch-size $INFERENCE_MAX_BATCH_SIZE
        --inference-max-seq-length $INFERENCE_MAX_SEQ_LENGTH
        "${MODEL_ARGS[@]}"
        )

echo "${megatron_arguments[@]}"
# Need to isntall these if not already installed
# pip install sqlitedict more-itertools sacrebleu evaluate pytablewriter

# TASKS="arc_easy,arc_challenge,piqa,hellaswag,openbookqa,mmlu,lambada_openai,winogrande,boolq,commonsense_qa"
TASKS="hellaswag"
NUM_FEWSHOT=0
echo "Launching evaluation with"
echo "${megatron_arguments[@]}"
echo "Tasks: $TASKS"
echo "Batch size: $BATCH_SIZE"
echo "Output file: ${OUTPUT_FILE}"
sbatch --nodes 4 launch_wrapper.sh \
    lm-evaluation-harness/lm_eval/__main__.py \
    --model megatron_lm \
    "${megatron_arguments[@]}" \
    --num_fewshot $NUM_FEWSHOT \
    --verbosity DEBUG \
    --log_samples \
    --tasks "$TASKS" \
    --batch_size $BATCH_SIZE \
    --output_path "${OUTPUT_FILE}"