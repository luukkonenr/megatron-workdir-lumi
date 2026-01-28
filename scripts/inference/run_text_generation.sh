#!/bin/bash


TOKENIZER_ARGS=(
    --tokenizer-model meta-llama/Llama-3.1-8B-Instruct
    --tokenizer-type HuggingFaceTokenizer
)

# MODEL_ARGS=(
#     --load /shared_silo/scratch/rluukkon/Megatron-Bridge/checkpoints/llama31-8b-bridge-test
#     --max-position-embeddings 131072
#     --encoder-seq-length 131072
#     --position-embedding-type rope
# )
source configs/llama3.1-8B.sh

MODEL_ARGS+=(
    --load /shared_silo/scratch/rluukkon/Megatron-Bridge/checkpoints/llama31-8b-bridge-test
    --use-checkpoint-args
    --use-mcore-models
    --micro-batch-size 8
    --attention-dropout 0.0
    --hidden-dropout 0.0

)
#   Lower values = less KV cache memory = can use larger batches
# - inference-batch-times-seqlen-threshold: Controls attention kernel selection
BATCH_SIZE=1024  # Increased from 16 - test and increase further if memory allows
INFERENCE_MAX_BATCH_SIZE=$((BATCH_SIZE * 2))  # Set higher than batch_size for safety
INFERENCE_MAX_SEQ_LENGTH=131072  # Most eval tasks don't need full 8192, saves memory


INFERENCE_SPECIFIC_ARGS=(
    --no-load-optim
    --no-load-rng
    --inference-max-batch-size $INFERENCE_MAX_BATCH_SIZE
    --inference-max-requests $INFERENCE_MAX_BATCH_SIZE
    --max-tokens-to-oom 100000
    --inference-max-seq-length $INFERENCE_MAX_SEQ_LENGTH
    # KV cache buffer size in GB - adjust based on available GPU VRAM
    # For optimal VRAM utilization, set this to use most of available VRAM
    # Formula: buffer_size_gb ≈ (Total_VRAM - Model_Weights - Overhead) * 0.7
    # For 80GB GPU with 8B model (~16GB weights): ~40-50GB is reasonable
    # For 40GB GPU with 8B model: ~20-25GB is reasonable
    # Increase this to maximize VRAM usage, but leave some headroom for activations
    --inference-dynamic-batching-buffer-size-gb 150.0
    --inference-dynamic-batching-block-size 512
    --inference-dynamic-batching
    # Max tokens per forward pass - increase for longer sequences/bigger batches
    # Default is 16384, increase if you have VRAM headroom
    # --inference-dynamic-batching-max-tokens 16384

    # --num-tokens-to-generate 20
    # --max-batch-size 16
)

    # ${TRAINING_ARGS[@]} \
# script=Megatron-LM/examples/inference/gpt/gpt_static_inference.py
script="Megatron-LM/tools/run_text_generation_server.py"

# script="run_batched_text_generation_server.py"
# bash $1 $script \
sbatch --nodes 1 $1 $script \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]}
    #--prompts "prompt one " "sample prompt two" "sample prompt 3"