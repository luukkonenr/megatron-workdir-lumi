#!/bin/bash
#SBATCH --job-name=llama-seqlen-test
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=480G
#SBATCH --partition=amd-tw-verification
#SBATCH --time=0-00:05:00
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export WANDB_API_KEY=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 

# symlink logfile to latest.out and latest.err
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.out logs/latest.out
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.err logs/latest.err

# add emerging-optimizers to python path
export PYTHONPATH=$PYTHONPATH:Emerging-Optimizers

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
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
SEQ_LENGTH=${SEQ_LENGTH:-8192} # 16384 # 32768 # 65536
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-0}

# With MOE, we need to set the number of experts
NUM_EXPERTS=${NUM_EXPERTS:-64}
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-8}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.001}
OPTIMIZER=${OPTIMIZER:-adam}
# Wandb
WANDB_ENTITY="risto-m-luukkonen"
WANDB_SAVE_DIR="$OUTPUT_DIR/wandb"
WANDB_PROJECT="moe_sweep_init_grid"
WANDB_EXP_NAME="llama3.1-8B-GBS${GLOBAL_BATCH_SIZE}-MRT${MOE_ROUTER_TOPK}-LR${LR}-NE${NUM_EXPERTS}-${SLURM_JOB_NAME}-${SLURM_JOBID}"


megatron_path=NVIDIA-Megatron-LM
# take model config from command line
MODEL_CONFIG=${1:-configs/llama3.1-8B.sh}


# TRAIN_DATA_CONFIG=configs/moe_scaling_sweep/training_args.sh

source environment.sh
source $MODEL_CONFIG
# source $TRAIN_DATA_CONFIG
# CONTAINER="../../containers/primus_v26.1.sif"
CONTAINER="/shared_silo/scratch/containers/rocm_primus_v25.11_transformers-4.5.7_linear_FA.sif"
# read args from arrays $MODEL_ARGS and $TRAINING_ARGS
echo "ARGS: ${MODEL_ARGS[*]} ${TRAINING_ARGS[*]} ${DATA_ARGS[*]} ${OUTPUT_ARGS[*]}"
echo "START $SLURM_JOBID: $(date)"

# echo "Running training..." \
srun --label \
    apptainer exec --rocm --overlay "/shared_silo/scratch/rluukkon/overlays/torchtitan-primus-26.1.img:ro" \
    "$CONTAINER" \
    "$LAUNCH_SCRIPT" \
    "${megatron_path}/pretrain_gpt.py" \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    --exit-interval 10


echo "END $SLURM_JOBID: $(date)"
