#!/bin/bash

source configs/llama3.1-8B.sh
MODEL_ARGS+=(
    --load /shared_silo/scratch/rluukkon/Megatron-Bridge/checkpoints/llama31-8b-bridge-test
)
# RANDOM_DIR="/tmp/lm_eval_$(date +%s%N)"
timestamp=$(date +%s)
OUTPUT_DIR="eval_results/"
OUTPUT_FILE="${OUTPUT_DIR}/run_${timestamp}"
# mkdir -p "$RANDOM_DIR"
# echo Saving temporary results to $RANDOM_DIR
mkdir -p $OUTPUT_DIR
echo Final results will be saved to: $OUTPUT_FILE

# Adding lm-evaluation-harness to PYTHONPATH without installing it for dev purposes
export PYTHONPATH=$PYTHONPATH:lm-evaluation-harness
export PYTHONPATH=$PYTHONPATH:Megatron-LM
megatron_arguments=(
        --no-load-optim 
        --no-load-rng 
        --max-tokens-to-oom 40000
        --micro-batch-size 1
        "${MODEL_ARGS[@]}"
        )
        # --qk-layernorm
        # --rotary-base 500000

echo "${megatron_arguments[@]}"
# install sqlitedict if not already installed
# pip install sqlitedict more-itertools sacrebleu evaluate pytablewriter
pip list
exit

# TASKS="arc_easy,arc_challenge,piqa,hellaswag,openbookqa,mmlu,lambada_openai,winogrande,boolq,commonsense_qa"
TASKS="hellaswag"
NUM_FEWSHOT=0
torchrun --nproc_per_node=8 lm-evaluation-harness/lm_eval/__main__.py \
    --model megatron_lm \
    "${megatron_arguments[@]}" \
    --num_fewshot $NUM_FEWSHOT \
    --verbosity DEBUG \
    --tasks "$TASKS" \
    --batch_size 16 \
    --output_path "${OUTPUT_FILE}"
