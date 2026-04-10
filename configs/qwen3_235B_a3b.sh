# H100 baseline config from https://github.com/yanring/Megatron-MoE-ModelZoo/blob/main/best_practice/qwen3/run_235b.sh
# A2A_OVERLAP=1 TP=2 PP=8 VPP=4 EP=32 NNODES=32 GBS=2048 bash ./sbatch_benchmarking.sh --recompute-granularity selective --recompute-modules moe_act layernorm --moe-router-force-load-balancing
#
# PP=8 with 94 layers: account-for-embedding + account-for-loss adds 2 virtual layers
# -> 94+2=96 total, 96/8=12 per stage, 12/4=3 per virtual stage (VPP=4)

MIN_LR=${MIN_LR:-0}
LR=${LR:-3e-4}
TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-8}
VPP_SIZE=${VPP_SIZE:-4}
HYBRIDEP_SMS=${HYBRIDEP_SMS:-16}
USE_DELAY_WGRAD=${USE_DELAY_WGRAD:-0}

# If TRAIN_ITERS is set externally, use it; otherwise default to a small number for testing
TRAIN_ITERS=${TRAIN_ITERS:-500}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-$TRAIN_ITERS}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-100}
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS:-$((LR_DECAY_ITERS / 2))}

# A2A overlap env vars (set by A2A_OVERLAP=1 in sbatch_benchmarking.sh)
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
# export NVTE_FWD_LAYERNORM_SM_MARGIN=24
# export NVTE_BWD_LAYERNORM_SM_MARGIN=24
# Recommended by reference YAML
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
# # GH200/JUPITER: CUDA Multicast is not supported for TP comm+GEMM overlap (UB); use CUDA IPC instead
# export UB_SKIPMC=1

MODEL_ARGS=(
    --num-layers 94
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 64
    --num-query-groups 4
    --group-query-attention
    --qk-layernorm
    --attention-softmax-in-fp32
    --max-position-embeddings 40960
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --kv-channels 128
    --disable-bias-linear
    --no-mmap-bin-files
    --init-method-std 0.006
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --bf16
    --swiglu
    --untie-embeddings-and-output-weights
    --num-experts 128
    --moe-router-topk 8
    --moe-ffn-hidden-size 1536
    --moe-router-load-balancing-type aux_loss
    --moe-router-force-load-balancing
    --moe-aux-loss-coeff 0.001
    --moe-z-loss-coeff 1e-4
    --moe-router-dtype fp32
    --expert-model-parallel-size 32
    --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type flex
    # --moe-flex-dispatcher-backend deepep
    # --moe-enable-deepep
    --moe-flex-dispatcher-backend hybridep
    --moe-hybridep-num-sms ${HYBRIDEP_SMS}
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-fusion
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size ${GLOBAL_BATCH_SIZE:-2048}
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --num-virtual-stages-per-pipeline-rank ${VPP_SIZE}
    --account-for-embedding-in-pipeline-split
    --account-for-loss-in-pipeline-split
    --sequence-parallel
    --context-parallel-size 1
    --use-distributed-optimizer
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr $LR
    --min-lr $MIN_LR
    --attention-backend fused
    --train-iters $TRAIN_ITERS
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr-decay-style WSD
    --lr-wsd-decay-style linear
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --clip-grad 1.0
    --weight-decay 0.05
    --make-vocab-size-divisible-by 256
    --dataloader-type single
    --num-workers 7
    --no-create-attention-mask-in-dataloader
    # --tp-comm-overlap is disabled on GH200/JUPITER: TransformerEngine Userbuffers
    # (both MC and CUDA IPC paths) are unreliable on this architecture. The IPC path
    # (UB_SKIPMC=1) hangs with stale push/receive counters (expecting N, got N-2).
    # --tp-comm-overlap
    --overlap-grad-reduce
    --overlap-param-gather
    --overlap-moe-expert-parallel-comm
    --recompute-granularity selective
    --recompute-modules moe_act layernorm
    --cross-entropy-loss-fusion
    --manual-gc
    --manual-gc-interval 5
    # --empty-unused-memory-level 1
)

if [ "${USE_DELAY_WGRAD}" = "1" ]; then
    TRAINING_ARGS+=(--delay-wgrad-compute)
fi

DATA_ARGS=(
    --mock-data
    --num-workers 7
    --num-dataset-builder-threads 7
    --split 99,1,0
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model qwen3-30b-a3b/
    --seq-length 4096
    --max-position-embeddings 4096
)

OUTPUT_ARGS=(
    # --save $CHECKPOINT_PATH
    # --save-interval 1000
    --ckpt-format torch_dist
    --auto-detect-ckpt-format
    --log-throughput
    --log-timers-to-tensorboard
    --log-progress
    --log-interval 1
    --eval-interval 100
    --eval-iters 10
    --exit-interval 10
)
