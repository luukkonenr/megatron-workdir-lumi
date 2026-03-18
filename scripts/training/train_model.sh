#!/bin/bash
#SBATCH --job-name=qwen3-30B-a3b
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=8
#SBATCH --mem=0G
#SBATCH --time=0-00:10:00
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
set -x

# add emerging-optimizers to python path
# export PYTHONPATH=$PYTHONPATH:Emerging-Optimizers

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

# These need to be before the source commands
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
SEQ_LENGTH=${SEQ_LENGTH:-8192}
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-0}
NUM_EXPERTS=${NUM_EXPERTS:-64}
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-8}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.001}
OPTIMIZER=${OPTIMIZER:-adam}
# Wandb
WANDB_ENTITY="risto-m-luukkonen"
WANDB_SAVE_DIR="$OUTPUT_DIR/wandb"
WANDB_PROJECT="moe_sweep_init_grid"
WANDB_EXP_NAME="350M-GBS${GLOBAL_BATCH_SIZE}-MRT${MOE_ROUTER_TOPK}-LR${LR}-NE${NUM_EXPERTS}-${SLURM_JOB_NAME}-${SLURM_JOBID}"


megatron_path=$PWD

# CONFIG_FILE can be passed as the first positional argument, via the CONFIG_FILE
# environment variable, or falls back to the default llama3.1-8B config.
CONFIG_FILE=${1:-${CONFIG_FILE:-configs/qwen3_30B_a3b.sh}}
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file not found: $CONFIG_FILE" >&2
    exit 1
fi
echo "Using model config: $CONFIG_FILE"

source $CONFIG_FILE
CONTAINER=${CONTAINER:-"/e/project1/laionize/luukkonen1/container_cachedir/nemo-v26.02-nemotron3-super.sif"}
# Default to 1; model configs requiring A2A overlap (A2A_OVERLAP=1) will override to 32
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# NCCL: use the HPC-X IB verbs plugin for JUPITER's Mellanox NDR200 fabric.
# This bypasses IPoIB (which has no IPv6 and causes errno 97 hangs) and uses
# native ibverbs/UCX for all inter-node collective traffic.
export NCCL_NET_PLUGIN=/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=ib

# NVSHMEM bootstrap for DeepEP RDMA path.
# PMI2 is available via Slurm's pmi2 plugin and works without MPI.
# export NVSHMEM_BOOTSTRAP=PMI2

# read args from arrays $MODEL_ARGS and $TRAINING_ARGS
echo "ARGS: ${MODEL_ARGS[*]} ${TRAINING_ARGS[*]} ${DATA_ARGS[*]} ${OUTPUT_ARGS[*]}"
echo "START $SLURM_JOBID: $(date)"

# echo "Running training..." \
srun --label \
    apptainer exec --nv --ipc \
    "$CONTAINER" \
    "$LAUNCH_SCRIPT" \
    "${megatron_path}/pretrain_gpt.py" \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}"

echo "END $SLURM_JOBID: $(date)"
