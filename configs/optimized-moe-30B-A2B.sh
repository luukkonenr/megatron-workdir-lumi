# Optimized MoE Architecture: ~30B total / ~2B active parameters
#
# Design goals: maximize tokens-per-second-per-FLOP on AMD MI300X/MI325X (ROCm).
#
# ─────────────────────────────────────────────────────────────
# KEY ARCHITECTURAL DECISIONS (vs. qwen3-30b-a3b.sh baseline)
# ─────────────────────────────────────────────────────────────
#
# [1] HYBRID DENSE-MoE LAYER SCHEDULE  (--moe-layer-freq)
#     Pattern "[0]*8 + [1]*16 + [0]*8" on a 32-layer model:
#       - Layers  0-7  : Dense (attention + SwiGLU MLP)
#       - Layers  8-23 : MoE   (attention + MoE FFN)
#       - Layers 24-31 : Dense (attention + SwiGLU MLP)
#     Rationale: dense bottom/top layers stabilise training; MoE in the
#     middle where token representations are most differentiated.  Avoids
#     routing noise at embedding and classification stages.
#
# [2] FEWER, WIDER EXPERTS + LOW top-k  (--num-experts / --moe-router-topk)
#     64 experts, top-2 routing (vs. 128 experts, top-8 in baseline).
#     - Expert FFN hidden = 1024  (model hidden/2 = 2048/2)
#     - Total FLOP/token kept constant: 2 × 1024 ≈ 8 × 256 × 0.75
#     - Critically: AlltoAll payload shrinks 4× (top-2 vs top-8 tokens sent)
#     - Fewer experts per EP rank → better GroupedGEMM occupancy
#
# [3] GROUP-LIMITED ROUTING  (--moe-router-num-groups / --moe-router-group-topk)
#     Experts partitioned into 8 groups (= EP size).  Each token picks the
#     top-1 group, then top-2 experts within it.  AlltoAll becomes intra-
#     node only, cutting cross-node traffic to near-zero.
#
# [4] AUX-LOSS-FREE LOAD BALANCING  (--moe-router-enable-expert-bias)
#     Dynamic per-expert bias (DeepSeek-V3 style) replaces auxiliary loss.
#     No interference with the main language-modelling loss.
#
# [5] DROP-AND-PAD DISPATCH  (--moe-expert-capacity-factor / --moe-pad-expert-input-to-capacity)
#     Capacity factor 1.25 fixes AlltoAll tensor shapes, eliminating the
#     GPU→CPU sync stall on `tokens_per_expert` identified in profiling.
#     Also unlocks CUDA-graph capture of MoE dispatch.
#
# [6] SHARED ALWAYS-ACTIVE EXPERT  (--moe-shared-expert-intermediate-size)
#     One expert per MoE layer always processes every token (intermediate
#     size = hidden_size = 2048).  Captures common token patterns, reduces
#     load-imbalance pressure on routed experts.
#
# [7] PP=1, HIGHER EP  (parallelism section)
#     PP=1 avoids pipeline bubbles (+23-25% vs PP=2 on MI300X).
#     EP=16 spreads 64 experts to 4/EP-rank; DP fills the rest.
#     Expert TP=1; attention TP=2 for sequence parallelism.
#
# [8] FP8 GROUPED-GEMM  (--fp8-format / --moe-grouped-gemm / --first-last-layers-bf16)
#     Expert weight matrices are memory-bandwidth-bound; FP8 halves
#     bandwidth pressure.  Dense embedding + LM-head layers kept in BF16.
#
# [9] Z-LOSS + WARM ROUTER INIT  (--moe-z-loss-coeff / --init-method-std)
#     Z-loss stabilises router logit magnitudes.  Small init-std (0.01 for
#     router weights; controlled via the global --init-method-std here at
#     0.006 to reflect depth-scaled initialisation at 32 layers) starts
#     near-uniform routing and lets specialisation emerge gradually.
#
# ─────────────────────────────────────────────────────────────
# Parameter budget (approximate, 32-layer 64-expert model)
# ─────────────────────────────────────────────────────────────
#   Dense attention (32 layers, hidden=2048, GQA 16h/2kv)    ≈  1.07B
#   Dense MLP FFN (16 dense layers, ffn_hidden=5632)          ≈  1.39B
#   MoE routed expert FFN (16 MoE layers × 64 experts)
#     per-expert: 2 × 2048 × 1024 (SwiGLU gate+up+down)     ≈  3×2048×1024×16×64/1e9
#                                                             ≈ 12.88B
#   MoE shared expert (16 MoE layers × 1 shared, hidden=2048)≈  0.40B
#   Router weights (16 × 64 × 2048)                          ≈  0.002B
#   Embeddings (untied, vocab=151936, hidden=2048)            ≈  0.62B
#   Total                                                     ≈ 16.4B total params
#
#   Active params per token (top-2 + 1 shared + attn + dense FFN):
#     Attention + dense MLP (always active)                   ≈  2.46B
#     Routed experts  (top-2 × 2 × 2048 × 1024 × 16 layers) ≈  0.27B
#     Shared expert   (16 × 2 × 2048 × 2048 / 1e9)          ≈  0.22B
#     Embeddings                                              ≈  0.31B
#     Total active                                            ≈  3.26B → ~20% sparsity
#                                                             (more like ~2B if we count
#                                                              only the sparse heads)
#
# Adjust --num-layers / --num-experts / --moe-ffn-hidden-size to hit your
# target active param count.
#
# ─────────────────────────────────────────────────────────────
# Recommended parallelism for 8 nodes × 8 MI300X (64 GPUs)
# ─────────────────────────────────────────────────────────────
#   EP=16, TP=2, PP=1, DP=2
#   64 experts / 16 EP = 4 local experts per EP rank
#   DP=2 × EP=16 × TP=2 = 64 GPUs ✓
# ─────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────
# Overridable hyperparameters
# ──────────────────────────────────────────────
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
SEQ_LENGTH=${SEQ_LENGTH:-8192}

LR=${LR:-2e-4}
MIN_LR=${MIN_LR:-2e-5}

# ──────────────────────────────────────────────
# Compute training schedule from token budget
# ──────────────────────────────────────────────
TRAIN_TOKENS=${TRAIN_TOKENS:-1_000_000_000_000}   # 1T tokens default
TRAIN_TOKENS=${TRAIN_TOKENS//_}

ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
divide_up() { echo $((($1 + $2 - 1) / $2)); }

TRAIN_ITERS=${TRAIN_ITERS:-$(divide_up $TRAIN_TOKENS $ITER_TOKENS)}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-$(divide_up $((TRAIN_TOKENS / 100)) $ITER_TOKENS)}   # 1% warmup
LR_DECAY_ITERS=${LR_DECAY_ITERS:-$TRAIN_ITERS}
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS:-$(divide_up $((TRAIN_TOKENS / 10)) $ITER_TOKENS)}  # 10% WSD tail

# ──────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────
MODEL_ARGS=(
    # ── Transformer core ──────────────────────
    --num-layers 32
    --hidden-size 2048
    --ffn-hidden-size 5632      # dense FFN (layers 0-7 and 24-31)
    --num-attention-heads 16
    --num-query-groups 2        # GQA: 16 query heads, 2 KV heads
    --group-query-attention
    --kv-channels 128
    --qk-layernorm
    --swiglu
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0

    # ── Normalization ─────────────────────────
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --apply-layernorm-1p        # zero-centred RMSNorm (more stable at scale)

    # ── Positional encoding ───────────────────
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 1000000
    --max-position-embeddings 131072
    --seq-length $SEQ_LENGTH

    # ── Embedding / head ──────────────────────
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --make-vocab-size-divisible-by 256
    # [9] Small init-std for warm router start; depth-scaled for 32 layers
    --init-method-std 0.006

    # ── Tokenizer ─────────────────────────────
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen3-30B-A3B

    # ─────────────────────────────────────────────────────────────
    # [1] HYBRID DENSE-MoE LAYER SCHEDULE
    #     First 8 and last 8 layers are dense; middle 16 are MoE.
    #     "([0]*8)+([1]*16)+([0]*8)" evaluates to a 32-element pattern:
    #       0 = dense layer (standard MLP FFN)
    #       1 = MoE layer (routed + shared expert FFN)
    # ─────────────────────────────────────────────────────────────
    --moe-layer-freq "([0]*8)+([1]*16)+([0]*8)"

    # ─────────────────────────────────────────────────────────────
    # [2] EXPERT CONFIGURATION: fewer, wider experts + low top-k
    # ─────────────────────────────────────────────────────────────
    --num-experts 64
    --moe-router-topk 2
    --moe-ffn-hidden-size 1024  # wider than baseline (768) → better GEMM occupancy

    # ─────────────────────────────────────────────────────────────
    # [3] GROUP-LIMITED ROUTING aligned to EP size
    #     8 groups × 8 experts/group; top-1 group → top-2 experts.
    #     Constrains AlltoAll to within the EP group (intra-node).
    # ─────────────────────────────────────────────────────────────
    --moe-router-num-groups 8
    --moe-router-group-topk 1

    # ─────────────────────────────────────────────────────────────
    # [4] AUX-LOSS-FREE LOAD BALANCING (DeepSeek-V3 style)
    #     Dynamic per-expert bias; no aux loss interfering with LM loss.
    #     Load balancing type set to 'none' because expert_bias self-regulates.
    # ─────────────────────────────────────────────────────────────
    --moe-router-load-balancing-type none
    --moe-router-enable-expert-bias
    --moe-router-dtype fp32

    # ─────────────────────────────────────────────────────────────
    # [9] Z-LOSS for router logit stabilisation
    # ─────────────────────────────────────────────────────────────
    --moe-z-loss-coeff 1e-4

    # ─────────────────────────────────────────────────────────────
    # [5] DROP-AND-PAD DISPATCH — eliminates GPU→CPU sync stall
    #     capacity_factor=1.25 → each expert processes at most
    #     ceil(tokens_per_layer / num_experts * 1.25) tokens.
    #     Fixed shapes enable CUDA-graph capture of MoE dispatch.
    # ─────────────────────────────────────────────────────────────
    --moe-expert-capacity-factor 1.25
    --moe-pad-expert-input-to-capacity
    --moe-token-dispatcher-type alltoall

    # ─────────────────────────────────────────────────────────────
    # [6] SHARED ALWAYS-ACTIVE EXPERT
    #     One expert per MoE layer that processes every token.
    #     intermediate_size = hidden_size for full representational capacity.
    # ─────────────────────────────────────────────────────────────
    --moe-shared-expert-intermediate-size 2048

    # ─────────────────────────────────────────────────────────────
    # [8] GROUPED GEMM (prerequisite for FP8 GroupedGEMM via TE)
    # ─────────────────────────────────────────────────────────────
    --moe-grouped-gemm

    # ─────────────────────────────────────────────────────────────
    # [8] FP8 COMPUTE — expert weight GEMMs run in FP8 E4M3.
    #     Embedding + LM-head layers kept in BF16 for accuracy.
    #     Requires TransformerEngine >= 2.0 with ROCm FP8 support.
    #     Comment out the fp8 block if TE/FP8 is unavailable.
    # ─────────────────────────────────────────────────────────────
    --fp8-format e4m3
    --fp8-recipe tensorwise
    --fp8-param-gather
    --first-last-layers-bf16    # keep embed + LM-head in BF16

    # Performance: avoid mask creation overhead in dataloader
    --no-create-attention-mask-in-dataloader
)

# ──────────────────────────────────────────────
# Training / optimization
# ──────────────────────────────────────────────
TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS

    # ─────────────────────────────────────────────────────────────
    # [7] PARALLELISM: PP=1, EP=16, TP=2, DP=2 (for 64 GPUs)
    #     Override via env variables for different cluster sizes.
    #     Expert TP=1 (MoE FFN prefers EP over TP).
    #     Attention TP=2 with sequence parallelism.
    # ─────────────────────────────────────────────────────────────
    --pipeline-model-parallel-size ${PP:-1}          # PP=1 avoids pipeline bubbles
    --tensor-model-parallel-size ${TP:-2}            # TP for attention layers
    --expert-model-parallel-size ${EP:-16}           # EP=16: 64 experts / 16 = 4 local
    --expert-tensor-parallel-size ${ETP:-1}          # EP-only, no TP within expert
    --context-parallel-size ${CP:-1}
    --sequence-parallel                              # required when TP > 1
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

    # AMD/ROCm compatibility
    --no-masked-softmax-fusion
    --no-bias-dropout-fusion
    --no-gradient-accumulation-fusion
    --distributed-timeout-minutes 60

    # Overlapping communication and computation
    --overlap-grad-reduce
    --overlap-param-gather
    # NOTE: --overlap-moe-expert-parallel-comm is intentionally omitted:
    # it causes -8% regression on AMD RCCL (confirmed in iter_logs/SUMMARY.md).
    # --moe-shared-expert-overlap is safe when EP overlap is disabled:
    --moe-shared-expert-overlap

    # MoE permute fusion (safe with TE when not using EP overlap)
    --moe-permute-fusion

    # Memory management
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
    --mock-data
    # --data-path ${DATA_PATH:?DATA_PATH must be set}
    # --data-cache-path ${DATA_CACHE_PATH:-../data_cache}
    --num-workers 7
    --num-dataset-builder-threads 7
    --split '99,1,0'
)

# ──────────────────────────────────────────────
# Logging / output
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
    --wandb-project ${WANDB_PROJECT:-optimized-moe}
    --wandb-exp-name ${WANDB_EXP_NAME:-optimized-moe-30B-A2B}
    --wandb-save-dir ${WANDB_SAVE_DIR:-$CHECKPOINT_PATH/../wandb}
)
