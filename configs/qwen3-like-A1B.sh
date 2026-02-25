# Qwen3-like MoE architecture with ~1B active parameters out of ~25B total.
#
# Design rationale:
#   This is a "scaled activation" sibling of Qwen3-30B-A3B. It keeps the same
#   hidden dimension (2048), KV head structure, and expert FFN size (768), but
#   reduces layers from 48 to 20 and top-k from 8 to 2. The result is a model
#   whose per-token compute cost matches a ~1B dense model while holding 25B
#   parameters in total, giving very high sparsity.
#
# Parameter budget (approximate):
#   Attention       :  20 layers × 10.49M params/layer         =   210M
#   All expert FFN  :  256 experts × 20 layers × 4.72M/expert  = 24,159M
#   Router weights  :  20 layers × 256 × 2048                  =    10M
#   Embeddings      :  2 × 151936 × 2048 (untied)              =   622M
#   Total           :                                           ≈ 25,001M (~25.0B)
#
# Active parameters per forward pass (top-k=2 experts per token):
#   Attention (always active)  :  210M
#   Active expert FFN (top-k=2):  2 × 20 × 4.72M              =   189M
#   Router                     :   10M
#   Embeddings                 :  622M
#   Total active               :                               ≈  1,031M (~1.03B)
#   Active ratio               :  1.03B / 25.0B               =    4.12%  (< 5% ✓)
#
# Recommended parallelism for 16 nodes × 8 MI325x (128 GPUs total):
#   EP=8  : 256 experts / 8 = 32 experts per GPU (within one node)
#   DP=16 : 128 GPUs / 8 EP = 16 data-parallel ranks across nodes
#   TP=1  : model fits comfortably on 1 GPU per DP×EP shard
#   PP=1  : no pipeline needed; all 20 layers fit in GPU memory easily
#
# Memory estimate per GPU with distributed optimizer (bf16, DP=16, EP=8):
#   Expert weights (sharded by EP=8)  :  24.16B / 8 × 2 bytes  ≈   6.0 GB
#   Attention + embed (replicated)    :  842M × 2 bytes         ≈   1.7 GB
#   Optimizer states (fp32, sharded by DP×EP=128):  25B×12B/128 ≈   2.3 GB
#   Activations (micro-batch=4, seq=8192):                      ≈   5.0 GB
#   Total estimated                                             ≈  15 GB  (of 256 GB)

MODEL_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen3-30B-A3B
    --num-layers 20
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 16
    --num-query-groups 4
    --group-query-attention
    --qk-layernorm
    --use-flash-attn
    --attention-softmax-in-fp32
    --max-position-embeddings 40960
    --seq-length 8192
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --disable-bias-linear
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --bf16
    --swiglu
    --kv-channels 128
    --untie-embeddings-and-output-weights
    --num-experts 256
    --moe-router-topk 2
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type global_aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-z-loss-coeff 1e-4
    --moe-router-dtype fp32
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --expert-model-parallel-size 8
    --vocab-size 151936
    --use-distributed-optimizer
)

# if train iterations is set, use it, otherwise use small number for testing
if [ -z "${TRAIN_ITERS:-}" ]; then
    TRAIN_ITERS=500
fi
if [ -z "${LR_DECAY_ITERS:-}" ]; then
    LR_DECAY_ITERS=500
fi
if [ -z "${LR_WSD_DECAY_ITERS:-}" ]; then
    LR_WSD_DECAY_ITERS=$((LR_DECAY_ITERS / 2))
fi

TRAINING_ARGS=(
    --micro-batch-size 8
    --global-batch-size 512
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --no-masked-softmax-fusion
    --no-bias-dropout-fusion
    --no-rope-fusion
    --distributed-timeout-minutes 60
    --overlap-grad-reduce
    --overlap-param-gather
    --use-distributed-optimizer
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr 3e-4
    --min-lr 0.0
    --train-iters $TRAIN_ITERS
    --lr-warmup-iters 100
    --lr-decay-style "WSD"
    --lr-wsd-decay-style "linear"
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --clip-grad 1.0
    --weight-decay 0.05
    # --tensorboard-dir $TENSORBOARD_DIR
    --log-throughput
    --log-timers-to-tensorboard
    --log-progress
    --log-interval 1
    --eval-interval 100
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --save-interval 1000
    --ckpt-format torch
    --auto-detect-ckpt-format
    --make-vocab-size-divisible-by 256
    --dataloader-type single
    --num-workers 7
)
