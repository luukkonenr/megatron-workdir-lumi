# Qwen3-Next-80B-A3B: hybrid MoE + Gated Delta Net architecture
#
# Reference: https://github.com/yanring/Megatron-MoE-ModelZoo/blob/main/model_configs/benchmarking/Qwen3-Next-80B-A3B.yaml
#
# Architecture highlights:
#   - 48-layer transformer with 512 experts (top-10 routing)
#   - Gated Delta Net (GDN) replaces 3/4 of attention layers (pattern: [1,1,1,0] repeating)
#   - Shared expert with gate
#   - Zero-Centered RMSNorm (--apply-layernorm-1p)
#   - Attention output gate on standard attention layers
#   - Multi-Token Prediction (MTP) with 1 additional layer
#   - GQA with 16 heads, 2 KV groups, kv_channels=256
#   - RoPE on first 25% of head dimensions, base=10M
#
# Active parameters per token: ~3B out of ~80B total

# ──────────────────────────────────────────────
# Overridable hyperparameters
# ──────────────────────────────────────────────
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
SEQ_LENGTH=${SEQ_LENGTH:-4096}

LR=${LR:-1.2e-4}
MIN_LR=${MIN_LR:-1.2e-5}

# ──────────────────────────────────────────────
# Compute training schedule from token budget
# ──────────────────────────────────────────────
TRAIN_TOKENS=${TRAIN_TOKENS:-30_000_000_000}
TRAIN_TOKENS=${TRAIN_TOKENS//_}

ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
divide_up() { echo $((($1 + $2 - 1) / $2)); }

TRAIN_ITERS=${TRAIN_ITERS:-$(divide_up $TRAIN_TOKENS $ITER_TOKENS)}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-$(divide_up $((TRAIN_TOKENS / 50)) $ITER_TOKENS)}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-$TRAIN_ITERS}
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS:-$(divide_up $((TRAIN_TOKENS / 5)) $ITER_TOKENS)}

# ──────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────
MODEL_ARGS=(
    # Transformer core
    --num-layers 48
    --hidden-size 2048
    --ffn-hidden-size 5120
    --num-attention-heads 16
    --num-query-groups 2
    --group-query-attention
    --kv-channels 256
    --qk-layernorm
    --attention-output-gate
    # --attention-backend fused
    --swiglu
    --bf16

    # FP8 compute (requires ROCm TransformerEngine fork)
    # --fp8-format e4m3
    # --fp8-recipe tensorwise
    # --fp8-param-gather
    # --moe-router-padding-for-quantization
    # --first-last-layers-bf16

    # Precision-aware optimizer (lower-precision optimizer states)
    # --use-precision-aware-optimizer
    # --exp-avg-dtype fp8
    # --exp-avg-sq-dtype fp8
    # --fp8-format e4m3
    # # --fp8-recipe tensorwise
    # --fp8-param-gather
    # --first-last-layers-bf16
    
    # Normalization
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --apply-layernorm-1p

    # Positional encoding
    --position-embedding-type rope
    --rotary-percent 0.25
    --rotary-base 10000000
    --max-position-embeddings 262144
    --seq-length $SEQ_LENGTH

    # Embedding / head
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --make-vocab-size-divisible-by 1187
    --init-method-std 0.02

    # Tokenizer
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen3-Next-80B-A3B-Instruct
    

    # Gated Delta Net (GDN) — replaces 3 out of every 4 attention layers
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq 4
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads 32

    # MoE
    --num-experts 512
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 512
    --moe-shared-expert-gate
    # --moe-shared-expert-overlap  # conflicts with --overlap-moe-expert-parallel-comm
    --moe-router-topk 10
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-3
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --moe-router-num-groups 8
    --moe-router-group-topk 2
    --moe-token-dispatcher-type alltoall

    # Multi-Token Prediction
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1

    # Perf: skip attention mask creation in dataloader (flash/TE attn creates it on-the-fly)
    --no-create-attention-mask-in-dataloader
)

# ──────────────────────────────────────────────
# Training / optimization
# ──────────────────────────────────────────────
TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS

    # Parallelism (override via env for multi-node)
    --tensor-model-parallel-size ${TP:-1}
    --pipeline-model-parallel-size ${PP:-1}
    # --use-megatron-fsdp # Gives segfault
    # --num-layers-per-virtual-pipeline-stage 6  # disabled: VPP requires PP>1
    --context-parallel-size ${CP:-1}
    --expert-model-parallel-size ${EP:-8}
    --use-distributed-optimizer

    # Optimizer
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr $LR
    --min-lr $MIN_LR
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr-decay-style WSD
    --lr-wsd-decay-style linear
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --clip-grad 1.0
    --weight-decay 0.1
    --apply-wd-to-qk-layernorm

    # Regularization
    --attention-dropout 0.0
    --hidden-dropout 0.0

    # AMD/ROCm compatibility flags
    --no-masked-softmax-fusion
    --no-bias-dropout-fusion
    --no-gradient-accumulation-fusion
    --distributed-timeout-minutes 60

    # recompute disabled — PP=1 gives enough memory headroom
    # (re-enable with smaller mbs or more EP to trade compute for memory)
    # --recompute-granularity selective
    # --recompute-modules core_attn shared_experts
    # --recompute-method block
    # --recompute-num-layers 12

    # Perf
    # --overlap-moe-expert-parallel-comm  # -8% regression on RCCL
    --overlap-grad-reduce
    --overlap-param-gather
    --moe-permute-fusion

    --manual-gc
    --manual-gc-interval 5

    # Data loading
    --dataloader-type single
    --num-workers 7

    # Checkpointing
    --ckpt-format torch_dist
    --auto-detect-ckpt-format
)

# ──────────────────────────────────────────────
# Data (override DATA_PATH before sourcing)
# ──────────────────────────────────────────────
DATA_ARGS=(
    # --data-path ${DATA_PATH:-/shared_silo/scratch/datasets/CC-MAIN-2020-high-actual/gpt-neox-20b/merged/CC-MAIN-2020-high-actual_merged}
    --mock-data
    # --data-cache-path ${DATA_CACHE_PATH:-../data_cache}
    --num-workers 7
    --num-dataset-builder-threads 7
    --split '99,1,0'
)

# ──────────────────────────────────────────────
# Logging / output  (expects $CHECKPOINT_PATH,
# $TENSORBOARD_DIR, $WANDB_* set by the trainer)
# ──────────────────────────────────────────────
OUTPUT_ARGS=(
    # --save ${CHECKPOINT_PATH:?CHECKPOINT_PATH must be set}
    # --save-interval 2000
    --eval-interval 500
    --eval-iters 2
    --log-throughput
    --log-timers-to-tensorboard
    --log-progress
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_DIR:-$CHECKPOINT_PATH/../tensorboard}
    --wandb-project ${WANDB_PROJECT:-qwen3-next}
    --wandb-exp-name ${WANDB_EXP_NAME:-qwen3-next-80B-A3B}
    --wandb-save-dir ${WANDB_SAVE_DIR:-$CHECKPOINT_PATH/../wandb}
)
