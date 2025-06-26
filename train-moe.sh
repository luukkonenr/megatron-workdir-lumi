#!/bin/bash
#SBATCH --job-name=test-moe
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=2
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000615
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -eox pipefail
echo "Starting bash script"
module purge
module load LUMI/24.03 partition/G

ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.out logs/latest.out
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.err logs/latest.err

DATASET="/scratch/project_462000353/data/nemotron-cc/tokenized-gemma-3/high-actual"
export HF_HOME="/scratch/project_462000353/hf_cache"
TOKENIZER_MODEL="google/gemma-3-27b-pt"
set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CC=gcc-12
export CXX=g++-12

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS #This is valid only if ntasks==ngpus
export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism

#OMP THREADING
export OMP_NUM_THREADS=1
export HSA_ENABLE_SDMA=0

# exp
EXPERIMENT="deepseek_v3"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH
echo "EXPERIMENT: $EXPERIMENT - $TIMESTAMP"
echo "MEGATRON_PATH: ${MEGATRON_PATH}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""

# network envs
# export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
# export TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY:-1}
# export NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE:-1}
# # NCCL_IB_HCA_LIST=$(rdma link -j | python3 -c "import sys, json; links=json.load(sys.stdin);names=[links[i]['ifname'] for i in range(8)]; print(*names,sep=',')")
# export NCCL_IB_HCA=${NCCL_IB_HCA:-$NCCL_IB_HCA_LIST}
# export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
# export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
# export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1} # Reducing to 1 ensures no PCIE traffic (even on single node)
# export NCCL_PROTO=${NCCL_PROTO:-Simple}
# export RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE:-0}
# export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0}

GPUS_PER_NODE=8
# cluster envs
RUN_ENV="${RUN_ENV:-slurm}"
# if [ $RUN_ENV = "cluster" ]; then
#     MASTER_ADDR=${MASTER_ADDR:-localhost}
#     MASTER_PORT=${MASTER_PORT:-$(shuf -n 1 -i 10000-65535)}
#     NNODES=${NNODES:-1}
#     NODE_RANK=${NODE_RANK:-0}
# elif [ $RUN_ENV = "slurm" ]; then
#     MASTER_ADDR=${SLURM_MASTER_ADDR}
#     MASTER_PORT=${SLURM_MASTER_PORT}
#     NNODES=$SLURM_NNODES
#     NODE_RANK=${SLURM_NODEID}
# fi

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo ""

if [ "${NNODES:-1}" -gt 1 ]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens51np0}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens51np0}"
    echo "NCCL and GLOO socket interfaces set."
else
    echo "Single node setup, skipping NCCL and GLOO socket interface settings."
fi

# model
MODEL_NAME="deepseek-ai/DeepSeek-V3"
MODEL_SIZE=${MODEL_SIZE:-tiny} # 16B, 236B, 671B.

# MODEL_SIZE=${MODEL_SIZE:-16B} # 16B, 236B, 671B.
# TOKENIZER_MODEL="deepseek-ai/DeepSeek-V3"
export HF_HOME=${HF_HOME:-"../huggingface"}
echo "MODEL_NAME: $MODEL_NAME"
echo "MODEL_SIZE: $MODEL_SIZE"
echo "TOKENIZER_MODEL: $TOKENIZER_MODEL"
echo "HF_HOME: $HF_HOME"
echo ""

# data

MOCK_DATA="${MOCK_DATA:-0}"

# For multi-node runs DATA_CACHE_PATH should point to a common path accessible by all the nodes (for eg, an NFS directory)
DATA_CACHE_PATH=${DATA_CACHE_PATH:-"../.cache"}

if [ "$MOCK_DATA" -eq 1 ]; then
    echo Using mock data.
    data_args="--mock-data --data-cache-path ${DATA_CACHE_PATH}"
else
    echo Using data from $DATASET
    data_args="--data-path ${DATASET}"
fi

# hyper params
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
SEQ_LEN="${SEQ_LEN:-2048}"
PAD_LEN=4096
PR=${PR:-bf16} # bf16, fp16, fp8
TP=${TP:-1}
ETP=${ETP:-1}
PP=${PP:-1}  
CP=${CP:-1}
EP=${EP:-1}
SP=true
DO=true
FL=true
SFT=false
FUSED_PADDED_MLA_ATTENTION=${FUSED_PADDED_MLA_ATTENTION:-false}
ATTENTION_SINK_K=${ATTENTION_SINK_K:-0}
WINDOW_SIZE=${WINDOW_SIZE:-none}
TRAIN_ITERS=${TRAIN_ITERS:-4000}
LR_WARMUP_ITERS=2
LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))
SAVE_INTERVAL=10
if [ "$ETP" -eq 1 ]; then
  if [ "$TP" != "$EP" ] && [ "$TP" -ne 1 ]; then
    echo "Error: TP and EP must be equal when ETP is 1."
    exit 1
  fi
fi
echo "TRAIN_ITERS: $TRAIN_ITERS"
echo "LR_WARMUP_ITERS: $LR_WARMUP_ITERS"
echo "LR_DECAY_ITERS: $LR_DECAY_ITERS"
echo ""

# perf options
OPTIMIZER_OFFLOAD=false
GEMM_TUNING="${GEMM_TUNING:-1}"
USE_GROUPED_GEMM="${USE_GROUPED_GEMM:-true}"
MOE_USE_LEGACY_GROUPED_GEMM="${MOE_USE_LEGACY_GROUPED_GEMM:-true}"
NVTE_CK_USES_BWD_V3="${NVTE_CK_USES_BWD_V3:-0}"
# NVTE_FUSED_ATTN=1
NVTE_FLASH_ATTN=1
GPT_LAYER_IN_TE="${GPT_LAYER_IN_TE:-true}"
echo "GEMM_TUING: $GEMM_TUNING"
echo "USE_GROUPED_GEMM: $USE_GROUPED_GEMM"
echo "MOE_USE_LEGACY_GROUPED_GEMM: $MOE_USE_LEGACY_GROUPED_GEMM"
echo "NVTE_CK_USES_BWD_V3: $NVTE_CK_USES_BWD_V3"
echo "GPT_LAYER_IN_TE: $GPT_LAYER_IN_TE"
echo ""

# memory options
AC="${AC:-none}" #none #sel #full
export RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-block} # block uniform
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
echo "AC: $AC"
echo "RECOMPUTE_METHOD: $RECOMPUTE_METHOD"
echo "RECOMPUTE_NUM_LAYERS: $RECOMPUTE_NUM_LAYERS"
echo ""

# do profile
PROFILE=${PROFILE:-false}
PROFILE_SYNC=${PROFILE_SYNC:-false}
PROFILE_START=${PROFILE_START:-6}
PROFILE_END=${PROFILE_END:-7}
FORCE_BALANCE=${FORCE_BALANCE:-false}
MOE_PERMUTE_FUSION=${MOE_PERMUTE_FUSION:-false}
echo "PROFILE: $PROFILE"
echo "PROFILE_START: $PROFILE_START"
echo "PROFILE_END: $PROFILE_END"
echo "FORCE_BALANCE: $FORCE_BALANCE"
echo ""

NAME="${RUN_ENV}-mcore-${MODEL_SIZE}-lr-${LR}-bs-${MICRO_BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-etp-${ETP}-ep-${EP}-ac-${AC}-do-${DO}-sp-${SP}-profile${PROFILE}-sync${PROFILE_SYNC}-${TIMESTAMP}"
OUTPUT_BASEPATH=output/${EXPERIMENT}-${NAME}

TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/"
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint"
TRAIN_LOG=${OUTPUT_BASEPATH}/log/${EXPERIMENT}-${NAME}.log
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
mkdir -p ${TENSORBOARD_DIR}
echo "OUTPUT_BASEPATH: $OUTPUT_BASEPATH"
echo "TENSORBOARD_DIR: $TENSORBOARD_DIR"
echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "TRAIN_LOG: $TRAIN_LOG"
echo ""
# PRETRAIN_CHECKPOINT_PATH="output/deepseek_v3-slurm-mcore-16B-lr-1e-5-bs-4-seqlen-2048-pr-bf16-tp-8-pp-1-etp-8-ep-1-ac-none-do-true-sp-true-profilefalse-syncfalse-20250517_010950/checkpoint/"

if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

# if [ "$NVTE_CK_USES_BWD_V3" -eq 1 ]; then
#     echo "Using BWD FAv3"
#     export NVTE_CK_USES_BWD_V3=1    #by default 0, if set to 1, some cases will call the bwd v3 dqdkdv kernel;
#     export NVTE_CK_V3_ATOMIC_FP32=0 #by default 1, if set to 0 will use atomic fp16/bf16(w/o convert_dq kernel) when NVTE_CK_USES_BWD_V3 is set to 1;
#     export NVTE_CK_V3_SPEC=1        #by default 0, if set to 1 will call the specialized v3 kernel when NVTE_CK_USES_BWD_V3 is set to 1;
# else
#     echo "Disabling BWD FAv3"
#     export NVTE_CK_USES_BWD_V3=0
#     export NVTE_CK_V3_ATOMIC_FP32=1
#     export NVTE_CK_V3_SPEC=0
# fi


# if [ -z ${MP_SFT_PACKING} ]; then
#     MP_SFT_PACKING=false
# fi


if [ $MODEL_SIZE = 671B ]; then
    HIDDEN_SIZE=7168 # change from 5120 to 7168
    NUM_ATTN_HEADS=128 
    NUM_LAYERS=${NUM_LAYERS:-61} # change from 60 to 61 layers
    INTERMEDIATE_SIZE=18432 # change from 12288 to 18432
    MOE_INTERMEDIATE_SIZE=2048 # change from 1536 to 2048
    MAX_POSITION_EMBEDDINGS=${SEQ_LEN:-4096} # change from 163840 to 163840
    EXTRA_VOCAB_SIZE=0 
    Q_LORA_RANK=1536
    KV_LORA_RANK=512
    QK_NOPE_HEAD_DIM=128
    QK_ROPE_HEAD_DIM=64
    V_HEAD_DIM=128
    ROPE_THETA=10000
    SCALE_FACTOR=40
    NUM_EXPERTS=256 # change from 160 to 256 experts
    ROUTER_TOPK=8 # change from 6 to 8 experts
    NUM_SHARED_EXPERTS=1 # 1 epxert shared
    # MOE_LAYER_FREQ='([0]*1+[1]*2)'  # 3 layer example
    MOE_LAYER_FREQ=1 # use '([0]*3+[1]*58)' for full model

    moe_options=" \
        --q-lora-rank ${Q_LORA_RANK} \
        --moe-router-score-function sigmoid \
        --moe-router-num-groups ${EP} \
        --moe-router-group-topk 4 \
        --moe-router-enable-expert-bias \
        --moe-router-bias-update-rate 1e-3
    "

elif [ $MODEL_SIZE = 236B ]; then
    HIDDEN_SIZE=5120
    NUM_ATTN_HEADS=128
    NUM_LAYERS=${NUM_LAYERS:-60}
    INTERMEDIATE_SIZE=12288
    MOE_INTERMEDIATE_SIZE=1536
    MAX_POSITION_EMBEDDINGS=${SEQ_LEN:-4096}
    EXTRA_VOCAB_SIZE=0
    Q_LORA_RANK=1536
    KV_LORA_RANK=512
    QK_NOPE_HEAD_DIM=128
    QK_ROPE_HEAD_DIM=64
    V_HEAD_DIM=128
    ROPE_THETA=10000
    SCALE_FACTOR=40
    NUM_EXPERTS=160
    ROUTER_TOPK=6
    NUM_SHARED_EXPERTS=2
    MOE_LAYER_FREQ=1

    moe_options=" \
        --q-lora-rank ${Q_LORA_RANK} \
        --moe-router-num-groups ${EP} \
        --moe-router-group-topk 3 \
    "

elif [ $MODEL_SIZE = 16B ]; then
    HIDDEN_SIZE=2048
    NUM_ATTN_HEADS=16
    NUM_LAYERS=${NUM_LAYERS:-24}
    INTERMEDIATE_SIZE=10944
    MOE_INTERMEDIATE_SIZE=1408
    MAX_POSITION_EMBEDDINGS=${SEQ_LEN:-4096}
    EXTRA_VOCAB_SIZE=0
    KV_LORA_RANK=512
    QK_NOPE_HEAD_DIM=128
    QK_ROPE_HEAD_DIM=64
    V_HEAD_DIM=128
    ROPE_THETA=10000
    SCALE_FACTOR=40
    NUM_EXPERTS=64
    ROUTER_TOPK=6
    NUM_SHARED_EXPERTS=2
    MOE_LAYER_FREQ=1
    
    moe_options=" \
        --moe-router-num-groups ${EP} \
        --moe-router-group-topk 1
    "
elif [ $MODEL_SIZE = tiny ]; then
    HIDDEN_SIZE=512
    NUM_ATTN_HEADS=8
    NUM_LAYERS=${NUM_LAYERS:-32}
    INTERMEDIATE_SIZE=2736
    MOE_INTERMEDIATE_SIZE=704
    MAX_POSITION_EMBEDDINGS=${SEQ_LEN:-4096}
    EXTRA_VOCAB_SIZE=0
    KV_LORA_RANK=128
    QK_NOPE_HEAD_DIM=32
    QK_ROPE_HEAD_DIM=16
    V_HEAD_DIM=32
    ROPE_THETA=10000
    SCALE_FACTOR=40
    NUM_EXPERTS=64
    ROUTER_TOPK=6
    NUM_SHARED_EXPERTS=2
    MOE_LAYER_FREQ=1
    
    moe_options=" \
        --moe-router-num-groups ${EP} \
        --moe-router-group-topk 1
    "
fi

if [ $MOE_PERMUTE_FUSION != false ]; then
    moe_permute_fustion_options=" \
            --moe-permute-fusion "
else
    moe_permute_fustion_options=""
fi
moe_options=" \
    --multi-latent-attention \
    ${moe_options} \
    --qk-layernorm \
    --attention-sink-k ${ATTENTION_SINK_K} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --num-shared-experts ${NUM_SHARED_EXPERTS} \
    --moe-shared-expert-intermediate-size  $((${NUM_SHARED_EXPERTS} * ${MOE_INTERMEDIATE_SIZE})) \
    --moe-router-load-balancing-type seq_aux_loss\
    --moe-router-topk ${ROUTER_TOPK} \
    ${moe_permute_fustion_options} \
    --num-experts ${NUM_EXPERTS} \
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --moe-aux-loss-coeff 1e-2 \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size ${ETP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --kv-channels ${V_HEAD_DIM} "
    

if [ $WINDOW_SIZE != none ]; then
    moe_options=" \
            ${moe_options} \
            --window-size ${WINDOW_SIZE}"
fi

if [ $FUSED_PADDED_MLA_ATTENTION = true ]; then
    moe_options=" \
            ${moe_options} \
            --fused-padded-mla-attention"
fi

# TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
TP_COMM_OVERLAP=0
comm_overlap_option="\
    --overlap-grad-reduce \
    --ddp-bucket-size 629145600 \
    --overlap-param-gather"

if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --tp-comm-overlap \
        --ddp-bucket-size 629145600 \
        --overlap-grad-reduce \
        --overlap-param-gather"
fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
		    --recompute-method ${RECOMPUTE_METHOD} \
		    --recompute-granularity full \
            --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
fi

if [ $USE_GROUPED_GEMM = true ]; then
    USE_GROUPED_GEMM_OPTION="--moe-grouped-gemm"
else
    USE_GROUPED_GEMM_OPTION=""
fi

if [ $MOE_USE_LEGACY_GROUPED_GEMM = true ]; then
    USE_LEGACY_GROUPED_GEMM_OPTION="--moe-use-legacy-grouped-gemm"
else
    USE_LEGACY_GROUPED_GEMM_OPTION=""
    # disable gemm tuning when using TE Group GEMM.
    GEMM_TUNING=0
    echo "[WARN] GEMM tuning is disabled when using TransformerEngine Group GEMM."
fi

# gemm tuning, https://github.com/ROCm/TransformerEngine
if [ "$GEMM_TUNING" -eq 1 ]; then
   export TE_HIPBLASLT_TUNING_RUN_COUNT=10
   export TE_HIPBLASLT_TUNING_ALGO_COUNT=50
else
   unset TE_HIPBLASLT_TUNING_RUN_COUNT
   unset TE_HIPBLASLT_TUNING_ALGO_COUNT
fi

if [ $GPT_LAYER_IN_TE = true ]; then
    TRANSFORMER_IMPL=transformer_engine
else
    TRANSFORMER_IMPL=local
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    TRANSFORMER_IMPL=transformer_engine
    pr_options=" \
        --bf16
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

if [ $OPTIMIZER_OFFLOAD = 'static' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy static \
        --optimizer-offload-fraction 1.0"
elif [ $OPTIMIZER_OFFLOAD = 'auto' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy auto"
else
    offload_option=""
fi

sft_option="--train-mode pretrain"

megatron_options="  \
	${data_args} \
	--log-throughput \
	--no-gradient-accumulation-fusion \
	--no-async-tensor-model-parallel-allreduce \
    --no-masked-softmax-fusion \
    --no-bias-dropout-fusion \
    --no-rope-fusion \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --dataloader-type cyclic \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --eval-interval 10000 \
        --eval-iters -1 \
        --save-interval ${SAVE_INTERVAL} \
        --ckpt-format torch \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-workers 2 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL}\
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon 1e-06 \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --rotary-scaling-factor ${SCALE_FACTOR} \
        --transformer-impl ${TRANSFORMER_IMPL} \
        $USE_GROUPED_GEMM_OPTION \
        $USE_LEGACY_GROUPED_GEMM_OPTION \
        --distributed-timeout-minutes 60 \
        --save ${CHECKPOINT_PATH} \
        --eod-mask-loss \
        --exit-interval 10
        "

if [ $PROFILE = true ]; then
    # blocking kernel
    if [ $PROFILE_SYNC = true ]; then
        export HIP_LAUNCH_BLOCKING=1
    fi

    profile_options=" \
            --profile \
            --profile-ranks 0 \
            --use-pytorch-profiler \
            --profile-step-start ${PROFILE_START} \
            --profile-step-end ${PROFILE_END} \
            --moe-router-force-load-balancing
            "
elif [ $FORCE_BALANCE = true ]; then
    profile_options=" \
            --moe-router-force-load-balancing
        "
else
    profile_options=" \
        "
fi

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS="--wandb-project=deepseek \
        --wandb-exp-name=LLama_${MODEL_SIZE} \
        --wandb-save-dir logs/wandb \
    "
else
   LOGGING_ARGS=""
fi



c="fe"
# Bind mask for one thread per core
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

echo "START $SLURM_JOBID: $(date)"
echo "NNODES" $SLURM_NNODES
echo "CPUS PER TASK" $SLURM_CPUS_PER_TASK

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif

export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
export PWD=(`pwd -P`)

# Avoid conflicts with $HOME/.local
export PYTHONUSERBASE=""

launcher="$PWD/launcher.sh"

run_cmd="Megatron-LM/pretrain_deepseek_v3.py 
 ${megatron_options} ${pr_options} ${load_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${moe_options} ${offload_option} ${comm_overlap_option} ${sft_option} ${vp_options} ${flash_options} ${profile_options} ${LOGGING_ARGS}"

echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    $launcher \
    $run_cmd


echo "END $SLURM_JOBID: $(date)"

singularity exec -B $SINGULARITY_BIND $CONTAINER python3 tools/throughput.py logs/${SLURM_JOB_NAME}-${SLURM_JOBID}.out