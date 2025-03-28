#!/bin/bash

##SBATCH --job-name=conv
##SBATCH --nodes=1
##SBATCH --mem=0
##SBATCH --partition=dev-g
##SBATCH --time=00-00:30:00
##SBATCH --gpus-per-node=mi250:1
##SBATCH --account=project_462000353
##SBATCH --output=logs/%j.out
##SBATCH --error=logs/%j.err

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif

CHECKPOINT_PATH=$1
OUTPUT_ROOT="./"
OUTPUT_DIR=$OUTPUT_ROOT/test_$(basename ${CHECKPOINT_PATH})_bfloat16
TOKENIZER=/scratch/project_462000353/models/llama31-8b/

python3 Megatron-LM/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader mcore \
  --saver llama_mistral \
  --tokenizer-dir ${TOKENIZER} \
  --load-dir ${CHECKPOINT_PATH} \
  --save-dir ${OUTPUT_DIR}