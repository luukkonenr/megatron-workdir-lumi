#!/bin/bash
#SBATCH --job-name=hf-meg-conv
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --partition=dev-g
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000963
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err  

# CONTAINER=/pfs/lustrep2/scratch/project_462000353/risto/containers/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-tev.2.2.0dev.sif
CONTAINER=/scratch/project_462000394/containers/for-turkunlp-team/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1-te-dockerhash-560b76ceab56.sif

HF_FORMAT_DIR=/scratch/project_462000353/models/llama31-8b
TOKENIZER_MODEL=$HF_FORMAT_DIR
TARGET_PP=1
TARGET_TP=2
MEGATRON_FORMAT_DIR=megatron-checkpoints/llama3.1-8B-TP-$TARGET_TP-PP-$TARGET_PP-test
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
export CUDA_DEVICE_MAX_CONNECTIONS=1

export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism
export CC=gcc-12
export CXX=g++-12
MEGATARON_PATH="Megatron-LM-new"
export PYTHONPATH=$PYTHONPATH:$MEGATARON_PATH

# This is a workaround for the issue with the container not having the correct version of transformers
# so we need to install it to userspace
singularity exec $CONTAINER /bin/bash -c "pip install transformers==4.48.2;
  python3 ${MEGATARON_PATH}/tools/checkpoint/convert.py \
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