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
overlay="/shared_silo/scratch/rluukkon/overlays/meg_eval.img"
overlay="$overlay:ro"
echo "Running: python3 lm-evaluation-harness/lm_eval/__main__.py $@"
args=$@

srun --label \
    apptainer exec --overlay $overlay $CONTAINER \
    bash -c "
    export RANK=$SLURM_PROCID;
    export LOCAL_RANK=$SLURM_LOCALID;
    torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --nnodes \$SLURM_JOB_NUM_NODES \
        --node_rank \$SLURM_NODEID \
        --rdzv_id \$SLURM_JOB_ID \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        $args
    "