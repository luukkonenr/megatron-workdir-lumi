#!/bin/bash
#SBATCH --job-name=test-rocm-6.2-new-cxi
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=2
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000353
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -eox pipefail
echo "Starting bash script"
module purge
module load LUMI/24.03 partition/G

#SET PATHS
BASE_DIR=(`pwd`)
EXPERIMENT_DIR="$BASE_DIR/workdir"

DATA_ROOT="/scratch/project_462000353/data/processed-llama31/merged"
CACHE_PATH="${DATA_ROOT}/index-cache"
DATA_PATH="0.7 ${DATA_ROOT}/fi-culturax 0.25 ${DATA_ROOT}/fineweb-edu-deduplicated 0.04 ${DATA_ROOT}/starcoder 0.01 ${DATA_ROOT}/xling"

CHECKPOINT_PATH="$EXPERIMENT_DIR/checkpoints"
TENSORBOARD_PATH="$EXPERIMENT_DIR/tensorboard/$SLURM_JOB_NAME-$SLURM_JOBID"
TOKENIZER_MODEL="/scratch/project_462000353/models/llama31-8b"

ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.out logs/latest.out
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.err logs/latest.err

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CC=gcc-12
export CXX=g++-12

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS #This is valid only if ntasks==ngpus
export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism

#OMP THREADING
export OMP_NUM_THREADS=2
export HSA_ENABLE_SDMA=0

#OMP THREADING
export OMP_NUM_THREADS=1
#export HSA_ENABLE_SDMA=1

#DEBUGGING, INCREASE VERBOSITY IN LOGS
# export MIOPEN_ENABLE_LOGGING=1
export PYTHONWARNINGS=ignore
# export TORCH_SHOW_CPP_STACKTRACES=1 
# export NCCL_DEBUG=INFO
# export RCCL_KERNEL_COLL_TRACE_ENABLE=1 
# export NCCL_DEBUG_SUBSYS=ALL 
# export NCCL_DEBUG_FILE=$EXPERIMENT_DIR/nccl-debug-${SLURM_JOB_NAME}-${SLURM_JOBID}.log #Move verbose nccl logging to its own file

#TransformerEngine
export NVTE_FLASH_ATTN=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_ROCM_ARCH=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
# export TORCH_LOGS="+dynamo" 
# export TORCHDYNAMO_VERBOSE=1

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

MODEL_SIZE="${MODEL_SIZE:-7B}"
FSDP="${FSDP:-0}"
TOTAL_ITERS="${TOTAL_ITERS:-100000}"
SEQ_LEN="${SEQ_LEN:-8192}"
GBS="${GBS:-128}"
MBS="${MBS:-1}"
RECOMPUTATION="${RECOMPUTATION:-0}"

# PARALLEL ARGS
PP="${PP:-1}"
TP="${TP:-1}"
CP_SIZE="${CP_SIZE:-1}"
VPP="${VPP:-1}"
USE_VPP="${USE_VPP:-0}"
CKPT_PATH="${CKPT_PATH:-None}"
PROFILE="${PROFILE:-0}"

#SAVING AND EVAL
LOG_INTERVAL=1
SAVE_INTERVAL=1000
EVAL_INTERVAL=10000
EVAL_STEPS=100

# https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json
if [[ $MODEL_SIZE = "200M" ]]; then #test
    NHIDDEN=576
    FFN_HIDDEN_SIZE=1536
    NLAYERS=30
    NHEADS=9
    NUM_KV_HEADS=3
    TIE_WORD_EMBEDDINGS=1
    
# https://huggingface.co/HuggingFaceTB/SmolLM-360M/blob/main/config.json
elif [ "$MODEL_SIZE" = "360M" ]; then
    NHIDDEN=960
    FFN_HIDDEN_SIZE=2560
    NLAYERS=32
    NHEADS=15
    NUM_KV_HEADS=5
    TIE_WORD_EMBEDDINGS=1

# https://huggingface.co/HuggingFaceTB/SmolLM-1.7B/blob/main/config.json
elif [ "$MODEL_SIZE" = "1.7B" ]; then 
    NHIDDEN=2048
    FFN_HIDDEN_SIZE=8192
    NLAYERS=24
    NHEADS=32
    NUM_KV_HEADS=32
    TIE_WORD_EMBEDDINGS=1

elif [ "$MODEL_SIZE" = "7B" ]; then
    NHIDDEN=4096
    FFN_HIDDEN_SIZE=14336
    NLAYERS=32
    NHEADS=32
    NUM_KV_HEADS=8
    TIE_WORD_EMBEDDINGS=0

elif [ "$MODEL_SIZE" = "33B" ]; then
    NHIDDEN=7168
    FFN_HIDDEN_SIZE=20480
    NLAYERS=56
    NHEADS=56
    NUM_KV_HEADS=8
    TIE_WORD_EMBEDDINGS=0
else
    echo "Unknown model size"
    exit 1
fi


GPT_ARGS="$GPT_ARGS --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
"
if [ "$NUM_KV_HEADS" != "$NHEADS" ]; then
    GPT_ARGS="$GPT_ARGS \
    --group-query-attention \
    --num-query-groups $NUM_KV_HEADS \
    "
fi

if [ "$TIE_EMBEDDINGS" = "0" ]; then
    GPT_ARGS="$GPT_ARGS --untie-embeddings-and-output-weights \
    "
fi

if [ "$FSDP" = "1" ]; then
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --use-torch-fsdp2 \
    "
else
PARALLEL_ARGS="$PARALLEL_ARGS \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --context-parallel-size $CP_SIZE \
    --sequence-parallel \
    --use-distributed-optimizer \
    "
fi

#TRAINING ARGS
#PYTORCH PROFILER ARGS
if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS="--use-pytorch-profiler --profile-ranks 0 --profile-step-start 5 --profile-step-end 7"
else
    PROFILE_ARGS=""
fi

GPT_ARGS="$GPT_ARGS \
    --attention-softmax-in-fp32 \
    --max-position-embeddings $SEQ_LEN \
    --use-flash-attn \
    --seq-length $SEQ_LEN \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --train-iters $TOTAL_ITERS \
    --bf16 \
    --swiglu \
    --no-async-tensor-model-parallel-allreduce \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --no-bias-dropout-fusion \
    --no-rope-fusion \
    --no-load-optim \
    --no-load-rng \
    --distributed-timeout-minutes 30 \
    --overlap-grad-reduce \
    "

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --ckpt-format torch \
    --lr 1e-4  \
    --min-lr 1e-8 \
    --lr-decay-style cosine \
    --lr-decay-iters $TOTAL_ITERS \
    --clip-grad 1.0 \
    --weight-decay 1.0e-1 \
    --lr-warmup-fraction .005 \
    "
TENSORBOARD_PATH="tensorboard/$SLURM_JOB_NAME"
OUTPUT_ARGS=" \
    --async-save \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_STEPS \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-throughput \
    --log-progress \
    --no-save-optim \
    --log-interval $LOG_INTERVAL \
    "

DATA_ARGS="
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --dataloader-type single \
    --save-interval 1000 \
    --eval-interval 10000 \
    --eval-iters 10 \
    --num-workers 2 \
    --data-path $DATA_PATH \
"
if [ "$USE_VPP" = "1" ]; then
    PARALLEL_ARGS="$PARALLEL_ARGS \
    --num-layers-per-virtual-pipeline-stage $VPP"
fi
if [ "$RECOMPUTATION" = "1" ]; then
    GPT_ARGS="$GPT_ARGS --recompute-activations --recompute-granularity selective"
fi
    # --load $CKPT_PATH \
CHECKPOINT_ARGS=""
CPKT_INTERVAL=1000
if [ "$CKPT_PATH" != "None" ]; then
    CHECKPOINT_ARGS="
    --save $CKPT_PATH \
    --save-interval $CPKT_INTERVAL \
    "
fi
CMD=" \
    Megatron-LM/pretrain_gpt.py \
    $GPT_ARGS \
    $OPTIMIZER_ARGS \
    $PARALLEL_ARGS \
    $CHECKPOINT_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    $PROFILE_ARGS \
    --exit-interval 20 \
    "
echo '============='
echo $CMD
echo '============='


c="fe"

# Bind mask for one thread per core
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

echo "START $SLURM_JOBID: $(date)"
echo "NNODES" $SLURM_NNODES
echo "CPUS PER TASK" $SLURM_CPUS_PER_TASK

CONTAINER_BASE=/scratch/project_462000394/containers/for-turkunlp-team/lumi/
CONTAINER_ID=lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-0fb1415058b3.sif
CONTAINER=${CONTAINER_BASE}${CONTAINER_ID}

export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/shs-libcxi-install/lib:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/libfabric-install-master/lib:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/aws-ofi-rccl2-install:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team/deps-2025-02-24/rccl-2025-02-26-85eb1f1/lib:\$LD_LIBRARY_PATH

export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
export PWD=(`pwd -P`)

# install libraries
export PYTHONUSERBASE="pythonuserbase"
singularity exec -B $PWD $CONTAINER bash -c "\$WITH_CONDA; pip install -U tensorboard; pip install /scratch/project_462000394/containers/for-turkunlp-team/flash_attn-2.7.3-cp312-cp312-linux_x86_64.whl /scratch/project_462000394/containers/for-turkunlp-team/transformer_engine-1.11.0+e7a7f6d-cp312-cp312-linux_x86_64.whl"


launcher="$PWD/launcher.sh"
if [ "$SLURM_JOB_PARTITION" = "dev-g" ]; then
    echo "Lumi dev-g partition is used, CPU binding is not used"
    srun --label \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    $launcher \
    $CMD
else
  echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
    srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B $PWD \
    -B $SINGULARITY_BIND \
    $CONTAINER \
    $launcher \
    $CMD
fi


echo "END $SLURM_JOBID: $(date)"