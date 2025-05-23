#!/bin/bash
#SBATCH --job-name=test-moe
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=64
#SBATCH --mem=480G
#SBATCH --partition=standard-g
#SBATCH --time=48:00:00
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
data_args=" \
    --data-path ${DATASET} \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    "

set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CC=gcc-12
export CXX=g++-12

#DISTRIBUTED ARGS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS #This is valid only if ntasks==ngpus
export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism
export NVTE_FLASH_ATTN=1

# All parameters from https://arxiv.org/pdf/2409.02060

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
LR=${LR:-4e-4}
MIN_LR=${MIN_LR:-4e-5}
SEQ_LEN="${SEQ_LEN:-4096}"
PAD_LEN=4096
TP=${TP:-1}
ETP=${ETP:-1}
PP=${PP:-2}
CP=${CP:-1}
EP=${EP:-1}
SP=true
DO=true
FL=true
SFT=false
ATTENTION_SINK_K=${ATTENTION_SINK_K:-0}
WINDOW_SIZE=${WINDOW_SIZE:-none}
TRAIN_ITERS=23842 # 100B tokens
LR_WARMUP_ITERS=200 # 2000 in olmoe
LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))
SAVE_INTERVAL=2385
CHECKPOINT_PATH=checkpoint-${SLURM_JOBID}

#OMP THREADING
export OMP_NUM_THREADS=1
export HSA_ENABLE_SDMA=0

MODEL_SIZE=${MODEL_SIZE:-"1B-7B"} 

if [ $MODEL_SIZE =  "1B-7B" ]; then
    HIDDEN_SIZE=2048
    INTERMEDIATE_SIZE=1024
    NUM_ATTN_HEADS=16
    NUM_LAYERS=${NUM_LAYERS:-16}
    MOE_INTERMEDIATE_SIZE=$INTERMEDIATE_SIZE
    MAX_POSITION_EMBEDDINGS=${SEQ_LEN:-4096}
    ROPE_THETA=10000
    SCALE_FACTOR=1 # ROTARY SCALING FACTOR
    NUM_EXPERTS=64
    ROUTER_TOPK=8
    MOE_LAYER_FREQ=1
    
    moe_options=" \
        --moe-router-num-groups ${EP} \
        --moe-router-group-topk 1
    "
fi
if [ $MODEL_SIZE =  "tiny" ]; then
    HIDDEN_SIZE=2048
    INTERMEDIATE_SIZE=1024
    NUM_ATTN_HEADS=16
    NUM_LAYERS=${NUM_LAYERS:-16}
    MOE_INTERMEDIATE_SIZE=$INTERMEDIATE_SIZE
    MAX_POSITION_EMBEDDINGS=${SEQ_LEN:-4096}
    EXTRA_VOCAB_SIZE=0
    # KV_LORA_RANK=128
    # QK_NOPE_HEAD_DIM=32*

    # QK_ROPE_HEAD_DIM=16
    # V_HEAD_DIM=32
    ROPE_THETA=10000
    SCALE_FACTOR=1 # ROTARY SCALING FACTOR
    NUM_EXPERTS=64
    ROUTER_TOPK=8
    # NUM_SHARED_EXPERTS=2
    MOE_LAYER_FREQ=1
    
    moe_options=" \
        --moe-router-num-groups ${EP} \
        --moe-router-group-topk 1
    "
fi



comm_overlap_option=" \
    --overlap-grad-reduce \
    --ddp-bucket-size 629145600 \
    --overlap-param-gather"

        # --tensorboard-queue-size 1 \
        # --tensorboard-dir ${TENSORBOARD_DIR} \
        # --log-timers-to-tensorboard \
        # --log-batch-size-to-tensorboard \
        # --log-validation-ppl-to-tensorboard \
        
        # --max-padding-length ${PAD_LEN} \
megatron_options="  \
	--use-flash-attn \
    --sequence-parallel \
    --context-parallel-size $CP \
    --use-distributed-optimizer \
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
        --init-method-std 0.02 \
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
        --log-interval 10 \
        --eval-interval 1000 \
        --eval-iters 50 \
        --save-interval ${SAVE_INTERVAL} \
        --ckpt-format torch \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-workers 2 \
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
        --transformer-impl transformer_engine \
        $USE_GROUPED_GEMM_OPTION \
        $USE_LEGACY_GROUPED_GEMM_OPTION \
        --distributed-timeout-minutes 60 \
        --save ${CHECKPOINT_PATH} \
        --bf16 \
        "
        # --eod-mask-loss \
        # --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    # --enable-shared-expert \
    # --num-shared-experts ${NUM_SHARED_EXPERTS} \

    # --moe-shared-expert-intermediate-size  $((${NUM_SHARED_EXPERTS} * ${MOE_INTERMEDIATE_SIZE})) \
moe_options=" \
    ${moe_options} \
    --moe-grouped-gemm \
    --qk-layernorm \
    --attention-sink-k ${ATTENTION_SINK_K} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk ${ROUTER_TOPK} \
    ${moe_permute_fustion_options} \
    --num-experts ${NUM_EXPERTS} \
    --moe-token-dispatcher-type alltoall \
    --moe-shared-expert-overlap \
    --moe-aux-loss-coeff 1e-2 \
    --moe-z-loss-coeff 1e-3 \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size ${ETP} \
    "
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
    # --kv-channels ${V_HEAD_DIM} "
    # --kv-lora-rank ${KV_LORA_RANK} \
    # --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    # --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    # --v-head-dim ${V_HEAD_DIM} \


arguments="${megatron_options} ${pr_options} ${load_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${moe_options} ${offload_option} ${comm_overlap_option} ${sft_option} ${vp_options} ${flash_options} ${profile_options} ${LOGGING_ARGS}"

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
program=Megatron-LM/pretrain_gpt.py

echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
srun --label --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B $PWD \
    $CONTAINER \
    $launcher \
    $program \
    $arguments


echo "END $SLURM_JOBID: $(date)"

singularity exec -B $SINGULARITY_BIND $CONTAINER python3 tools/throughput.py logs/${SLURM_JOB_NAME}-${SLURM_JOBID}.out