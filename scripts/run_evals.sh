#!/bin/bash
#SBATCH --job-name=eval-harness
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --partition=dev-g
#SBATCH --time=00:90:00
#SBATCH --account=project_462000353
#SBATCH --exclusive
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err


ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.out logs/latest.out
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.err logs/latest.err
export PWD=(`pwd -P`)
workdir=${PWD}
export PYTHONUSERBASE=".local"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS #This is valid only if ntasks==ngpus
export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism
export CC=gcc-12
export CXX=g++-12
# SINGULARITY 
CONTAINER=/pfs/lustrep2/scratch/project_462000353/risto/containers/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-tev.2.2.0dev.sif
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
CHECKPOINT_PATH=checkpoints/flame-moe-419m-12872205/
# CHECKPOINT_PATH=c"checkpoints/flame-moe-290m-12969781/"
        # --tokenizer-model $TOKENIZER_MODEL


# RANDOM_DIR="/tmp/lm_eval_$(date +%s%N)"
# timestamp=$(date +%s)
# OUTPUT_FILE="${OUTPUT_DIR}/run_${timestamp}"
# OUTPUT_DIR="eval_results_$SLURM_NTASKS_PER_NODE"
# mkdir -p "$RANDOM_DIR"
# echo Saving temporary results to $RANDOM_DIR
# mkdir -p $OUTPUT_DIR
# echo Final results will be saved to: $OUTPUT_FILE

# Adding lm-evaluation-harness to PYTHONPATH without installing it for dev purposes
export PYTHONPATH=$PYTHONPATH:lm-evaluation-harness
export PYTHONPATH=$PYTHONPATH:Megatron-LM
NUM_FEWSHOT=0
megatron_arguments=(--load $CHECKPOINT_PATH
        --no-load-optim 
        --no-load-rng 
        --max-tokens-to-oom 40000
        --use-checkpoint-args # use model args from checkpoint
        --use-mp-args-from-checkpoint-args # use model parallel args from checkpoint
        --micro-batch-size 1
        --bf16
        --use-flash-attn
        --tokenizer-type HuggingFaceTokenizer
        )
        # --rotary-base 500000

# # install sqlitedict if not already installed
# srun --label \
#     singularity exec \
#     -B ${PWD} \
#     $CONTAINER \
#     pip install --user sqlitedict more-itertools

srun --label \
    singularity exec \
    -B ${PWD} \
    $CONTAINER \
    ./launcher.sh \
    ${workdir}/lm-evaluation-harness/lm_eval/__main__.py \
    --model megatron_lm \
    "${megatron_arguments[@]}" \
    --num_fewshot $NUM_FEWSHOT \
    --verbosity DEBUG \
    --tasks hellaswag \
    --batch_size 16
    
#     # --output_path $RANDOM_DIR \
    
 
# echo Moving temporary results from $RANDOM_DIR to $OUTPUT_FILE
# find "$RANDOM_DIR" -name "results_*.json" -exec mv {} "$OUTPUT_FILE" \;
# rm -rf "$RANDOM_DIR"