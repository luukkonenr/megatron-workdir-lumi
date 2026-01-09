#!/bin/bash
#SBATCH --job-name=test
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --nodes=2
#SBATCH --mem=480G
#SBATCH --partition=amd-tw-verification
#SBATCH --time=0-01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source environment.sh
# launch given script with srun

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

# Set network interfaces for multi-node communication
# Gloo backend needs GLOO_SOCKET_IFNAME to avoid using loopback (127.0.1.1)
# NCCL backend uses NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=eno0
export NCCL_SOCKET_IFNAME=eno0

overlay="/shared_silo/scratch/rluukkon/overlays/meg_eval.img"
overlay="$overlay:ro"
echo "Running: python3 lm-evaluation-harness/lm_eval/__main__.py $@"
# ÷TORCHRUN_ARGS="--nproc_per_node=8 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
args=$@

srun --label \
    apptainer exec --overlay $overlay $CONTAINER \
    bash -c "
    export RANK=$SLURM_PROCID;
    export LOCAL_RANK=$SLURM_LOCALID;
    export MASTER_ADDR=$MASTER_ADDR;
    export MASTER_PORT=$MASTER_PORT;
    export GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME;
    export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME;
    torchrun --nproc_per_node=8 \
        --nnodes \$SLURM_JOB_NUM_NODES \
        --node_rank \$SLURM_NODEID \
        --rdzv_id \$SLURM_JOB_ID \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        lm-evaluation-harness/lm_eval/__main__.py $args
    "