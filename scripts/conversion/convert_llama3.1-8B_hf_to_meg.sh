#!/bin/bash

# HF_FORMAT_DIR=meta-llama/Llama-3.1-8B-Instruct
# TOKENIZER_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_FORMAT_DIR=/shared_silo/scratch/rluukkon/hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
TOKENIZER_MODEL=${HF_FORMAT_DIR}
TARGET_PP=1
TARGET_TP=1
MEGATRON_FORMAT_DIR=megatron-checkpoints/llama3.1-8B-TP-$TARGET_TP-PP-$TARGET_PP-test
MEGATARON_PATH="Megatron-LM"
export PYTHONPATH=$PYTHONPATH:$MEGATARON_PATH

# This is a workaround for the issue with the container not having the correct version of transformers
# so we need to install it to userspace
apptainer exec $CONTAINER /bin/bash -c " python3 ${MEGATARON_PATH}/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader llama_mistral \
    --model-size llama3 \
    --checkpoint-type 'hf' \
    --saver mcore \
    --target-tensor-parallel-size ${TARGET_TP} \
    --target-pipeline-parallel-size ${TARGET_PP} \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL}
   "