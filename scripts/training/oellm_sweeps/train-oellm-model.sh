#!/bin/bash
#SBATCH --job-name=350M
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --nodes=2
#SBATCH --mem=480G
#SBATCH --partition=amd-tw-verification
#SBATCH --time=0-01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export WANDB_API_KEY=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 


export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.out logs/latest.out
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.err logs/latest.err
BASE_DIR=$PWD

OUTPUT_DIR=$BASE_DIR/output/$SLURM_JOB_NAME-$SLURM_JOBID
CHECKPOINT_PATH=$OUTPUT_DIR/checkpoints

mkdir -p $CHECKPOINT_PATH $BASE_DIR/logs $OUTPUT_DIR

LAUNCH_SCRIPT="./launcher.sh"

TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard/$SLURM_JOB_NAME-$SLURM_JOBID
# export WANDB_API_KEY=e180e74ecbd5eb200b9a130666e5cb8e89b15327

# These need to be before the source commands
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-0}
NUM_EXPERTS=${NUM_EXPERTS:-64}
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-8}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.001}

# Wandb
WANDB_ENTITY="risto-m-luukkonen"
WANDB_SAVE_DIR="$OUTPUT_DIR/wandb"
WANDB_PROJECT="moe_sweep_test"
WANDB_EXP_NAME="350M-GBS${GLOBAL_BATCH_SIZE}-MRT${MOE_ROUTER_TOPK}-LR${LR}-NE${NUM_EXPERTS}-${SLURM_JOB_NAME}-${SLURM_JOBID}"


megatron_path=Megatron-LM
MODEL_CONFIG=configs/moe_scaling_sweep/models/flame-moe-100M-350M.sh
TRAIN_DATA_CONFIG=configs/moe_scaling_sweep/training_args.sh
source environment.sh
source $MODEL_CONFIG
source $TRAIN_DATA_CONFIG
CONTAINER="../../containers/primus_v26.1.sif"


# read args from arrays $MODEL_ARGS and $TRAINING_ARGS
echo "ARGS: ${MODEL_ARGS[*]} ${TRAINING_ARGS[*]} ${DATA_ARGS[*]} ${OUTPUT_ARGS[*]}"
echo "START $SLURM_JOBID: $(date)"
srun --label \
    apptainer exec --rocm \
    "$CONTAINER" \
    "$LAUNCH_SCRIPT" \
    "${megatron_path}/pretrain_gpt.py" \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}"

echo "END $SLURM_JOBID: $(date)"
