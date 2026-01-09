#!/bin/bash

TOKENIZER_ARGS=(
    --tokenizer-model meta-llama/Llama-3.1-8B-Instruct
    --tokenizer-type HuggingFaceTokenizer
)

# MODEL_ARGS=(
#     --use-checkpoint-args
#     --use-mcore-models
#     --load /shared_silo/scratch/rluukkon/Megatron-Bridge/checkpoints/llama31-8b-bridge-test
#     --max-position-embeddings 131072
#     --encoder-seq-length 131072
#     --position-embedding-type rope
# )
source configs/llama3.1-8B.sh
MODEL_ARGS+=(
    --load /shared_silo/scratch/rluukkon/Megatron-Bridge/checkpoints/llama31-8b-bridge-test
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --num-tokens-to-generate 20
    --max-batch-size 4
    --no-load-optim
    --no-load-rng
)


python3 Megatron-LM/examples/inference/gpt/gpt_static_inference.py \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]} \
    --prompts "prompt one " "sample prompt two" "sample prompt 3"