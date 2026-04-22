# Qwen3.5-35B-A3B-Base (text stack) for Megatron-LM
#
# Hugging Face reference: https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base/blob/main/config.json
# Layout inspired by: configs/qwen3-30b-a3b.sh (MoE + training defaults) and
# configs/qwen3.5-9B-text.sh / configs/qwen3-next-80B-A3B.sh (hybrid GDN + MTP).
#
# HF text_config mapping (multimodal / vision_* omitted):
#   num_hidden_layers=40, hidden_size=2048,
#   num_attention_heads=16, num_key_value_heads=2, head_dim=256,
#   moe_intermediate_size=512, shared_expert_intermediate_size=512,
#   num_experts=256, num_experts_per_tok=8,
#   layer_types / full_attention_interval=4 -> 3x gated_delta_net + 1x full SDPA per block of 4,
#   linear_* -> --linear-* below,
#   attn_output_gate=true -> --attention-output-gate,
#   rope_theta=1e7, partial_rotary_factor=0.25 -> --rotary-base / --rotary-percent,
#   max_position_embeddings=262144, rms_norm_eps=1e-6,
#   router_aux_loss_coef=0.001, vocab_size=248320, tie_word_embeddings=false,
#   mtp_num_hidden_layers=1 -> --mtp-num-layers 1 (mtp_use_dedicated_embeddings=false: Megatron default).
#
# Override parallelism / batching via env before sourcing, e.g. EP=8 SEQ_LENGTH=8192.

: "${SEQ_LENGTH:=8192}"
: "${EXPERT_MODEL_PARALLEL_SIZE:=8}"

MODEL_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen3.5-35B-A3B-Base
    --num-layers 40
    --hidden-size 2048
    --ffn-hidden-size 512
    --num-attention-heads 16
    --num-query-groups 2
    --group-query-attention
    --kv-channels 256
    --qk-layernorm
    --attention-output-gate
    --apply-wd-to-qk-layernorm
    --attention-backend auto
    --attention-softmax-in-fp32
    --max-position-embeddings 262144
    --seq-length "${SEQ_LENGTH}"
    --position-embedding-type rope
    --rotary-base 10000000
    --rotary-percent 0.25
    --disable-bias-linear
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --bf16
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 248320
    # Hybrid attention (HF layer_types: linear_attention x3 + full_attention x1 repeating)
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq 4
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads 32
    # MoE + shared expert (HF moe_intermediate_size / shared_expert_intermediate_size)
    --num-experts 256
    --moe-router-topk 8
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 512
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-router-dtype fp32
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --use-distributed-optimizer
    --no-create-attention-mask-in-dataloader
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
    --micro-batch-size 1
    --global-batch-size 16
    # --no-async-tensor-model-parallel-allreduce
    --no-masked-softmax-fusion
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    # --no-rope-fusion
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
    --expert-tensor-parallel-size 1
    --context-parallel-size 8
    --sequence-parallel
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
    --num-workers 2
)
