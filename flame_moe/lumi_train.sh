#!/bin/bash
#SBATCH --job-name=test-moe
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=dev-g
##SBATCH --partition=standard-g
#SBATCH --time=0-00:15:00
#SBATCH --exclusive
#SBATCH --account=project_462000353
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -eox pipefail
echo "Starting bash script"
module purge
module load LUMI/24.03 partition/G

ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.out logs/latest.out
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.err logs/latest.err

export HF_HOME="/scratch/project_462000353/hf_cache"
SAVE_DIR=${PWD}/checkpoints/${SLURM_JOB_NAME}-${SLURM_JOBID}

set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CC=gcc-12
export CXX=g++-12

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS #This is valid only if ntasks==ngpus
export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism
#TransformerEngine
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN_BACKEND=1
export NVTE_FUSED_ATTN_AOTRITON=1
# export NVTE_DEBUG=1
# export NVTE_DEBUG_LEVEL=3
# export NVTE_LOG_CK_CONFIG=1

export NVTE_ROCM_ARCH=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
# export TORCH_LOGS="+dynamo" 

#OMP THREADING
export OMP_NUM_THREADS=1
export HSA_ENABLE_SDMA=0


c="fe"

# 

#####################
### < BOILERPLATE > ###
# Bind mask for one thread per core
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

echo "START $SLURM_JOBID: $(date)"
echo "NNODES" $SLURM_NNODES
echo "CPUS PER TASK" $SLURM_CPUS_PER_TASK

CONTAINER=/pfs/lustrep2/scratch/project_462000353/risto/containers/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-tev.2.2.0dev.sif
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
export PWD=(`pwd -P`)

# Avoid conflicts with $HOME/.local
export PYTHONUSERBASE=""

launcher="$PWD/launcher.sh"
source flame_moe/configs/flame_moe.sh
#####################
### < BOILERPLATE /> ###

# program=Megatron-LM/pretrain_flame_moe.py
# program=Megatron-LM-nvidia/pretrain_gpt.py
program=Megatron-LM-IFU/pretrain_gpt.py

export ROCM_GPU_ARCH_OVERRIDE="gfx90a"  # or your actual arch like gfx908, gfx906, etc.
# export HIPBLASLT_FORCE_ENABLE_UNSUPPORTED=1
# export HIPBLASLT_DISABLE_HW_CHECK=1

echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    $launcher \
    $program \
    "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${MODEL_ARGS[@]}" "${SAVE_ARGS[@]}" "${INFRA_ARGS[@]}"


echo "END $SLURM_JOBID: $(date)"

singularity exec -B $SINGULARITY_BIND $CONTAINER python3 tools/throughput.py logs/${SLURM_JOB_NAME}-${SLURM_JOBID}.out