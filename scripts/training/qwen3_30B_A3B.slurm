#!/bin/bash
#SBATCH --job-name=test
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --nodes=4
#SBATCH --mem=0
#SBATCH --partition=amd-tw-verification
#SBATCH --time=0-00:10:00
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.out logs/latest.out
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.err logs/latest.err

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

BASE_DIR=/shared_silo/scratch/rluukkon/oellm/oellm_fsdp
OUTPUT_DIR=checkpoints/
CHECKPOINT_PATH=$OUTPUT_DIR/qwen3-30b-a3b-bridge-test
# TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard/$SLURM_JOB_NAME-$SLURM_JOBID

# Load OELLM checkpoint weights (fresh optimizer for new dataset)
# LOAD_CHECKPOINT_PATH=/shared_silo/scratch/dzautner/oellm_fsdp/checkpoints_tp1_with_optim
# LOAD_CHECKPOINT_PATH=/shared_silo/scratch/rluukkon/oellm/oellm_fsdp/checkpoints_tp1_with_optim_fixed

mkdir -p $CHECKPOINT_PATH $BASE_DIR/logs $OUTPUT_DIR

LAUNCH_SCRIPT="/shared_silo/scratch/rluukkon/oellm/oellm_fsdp/launch.sh"

# # Wandb
# WANDB_PROJECT="hplt2c_euro_4T_9B"
# WANDB_EXP_NAME="OELLM-9B-8020"
# WANDB_DIR="$OUTPUT_DIR/wandb"
# export WANDB_API_KEY=e180e74ecbd5eb200b9a130666e5cb8e89b15327
TRAIN_TOKENS=30_000_000_000   # TRAIN_ITERS computed from this

divide_rounding_up() {
    echo $((($1+$2-1)/$2))
}
MIN_LR=0
WARMUP_FRACTION=1/10
COOLDOWN_FRACTION=1/5
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=1  # Decreased from 4 to 2 with 9B model paramaters

SEQ_LENGTH=2048
# Calculate TRAIN_ITERS from TRAIN_TOKENS
TRAIN_TOKENS=${TRAIN_TOKENS//_}    # drop "_" for bash math
ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS=$(divide_rounding_up $TRAIN_TOKENS $ITER_TOKENS)

#Set LR_WARMUP_ITERS and LR_WSD_DECAY_ITERS based on WARMUP_FRACTION
#and COOLDOWN_FRACTION
WARMUP_CALC=$((TRAIN_ITERS * ${WARMUP_FRACTION}))

# Cap warmup iterations at 25000 to prevent excessively long warmup periods
if [ $WARMUP_CALC -gt 25000 ]; then
   LR_WARMUP_ITERS=25000
else
   LR_WARMUP_ITERS=$WARMUP_CALC
fi

LR_WSD_DECAY_ITERS=$((TRAIN_ITERS*${COOLDOWN_FRACTION}))

LR_DECAY_ITERS=$TRAIN_ITERS

megatron_path=Megatron-LM
source environment.sh
source configs/qwen3-30b-a3b.sh

DATA_ARGS=(
    # --data-path wikipedia_20220301.en.valid.jsonl.preprocessed_text_document
    # --data-cache-path ../data_cache
    --mock-data 
    --moe-router-force-load-balancing
)
# LOAD_ARGS=(
#     --load /shared_silo/scratch/rluukkon/Megatron-Bridge/checkpoints/qwen3-30b-a3b-bridge-test
#     --ckpt-format torch_dist
#     --no-load-optim
#     --no-load-rng
#     --finetune
# )
# No loading
LOAD_ARGS=()
export GEMM_TUNING=1
# read args from arrays $MODEL_ARGS and $TRAINING_ARGS
echo "ARGS: ${MODEL_ARGS[*]} ${TRAINING_ARGS[*]} ${DATA_ARGS[*]} ${LOAD_ARGS[*]}"

echo "START $SLURM_JOBID: $(date)"
srun --label \
    apptainer exec --rocm \
    "$CONTAINER" \
    "$LAUNCH_SCRIPT" \
    "${megatron_path}/pretrain_gpt.py" \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${LOAD_ARGS[@]}"

echo "END $SLURM_JOBID: $(date)"
