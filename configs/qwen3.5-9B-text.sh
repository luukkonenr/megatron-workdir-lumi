# Megatron-LM training config aligned with Hugging Face Qwen3.5-9B **text** config
# (https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json).
# Vision / multimodal settings are omitted.
#
# HF text_config mapping:
#   num_hidden_layers=32, hidden_size=4096, intermediate_size=12288,
#   num_attention_heads=16, num_key_value_heads=4, head_dim=256,
#   attn_output_gate=true -> --attention-output-gate (SDPA path),
#   full_attention_interval=4  -> 3 linear-attn (GDN) + 1 full SDPA per 4 layers,
#   linear_* -> gated_delta_net knobs below,
#   max_position_embeddings=262144, rms_norm_eps=1e-6, rope_theta=1e7,
#   partial_rotary_factor=0.25 -> --rotary-percent 0.25,
#   vocab_size=248320, tie_word_embeddings=false.
# HF also sets partial_rotary_factor=0.25 and mRoPE sections for multimodal;
# this file uses standard RoPE (--rotary-percent 1.0) for text-only Megatron.

MIN_LR=${MIN_LR:-0}
LR=${LR:-3e-4}

: "${GLOBAL_BATCH_SIZE:=2048}"
: "${SEQ_LENGTH:=8192}"

TRAIN_TOKENS=30_000_000_000
TRAIN_TOKENS=${TRAIN_TOKENS//_}
ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))

divide_rounding_up() {
    echo $((($1+$2-1)/$2))
}

TRAIN_ITERS=$(divide_rounding_up $TRAIN_TOKENS $ITER_TOKENS)
LR_WARMUP_ITERS=$(divide_rounding_up $((TRAIN_TOKENS / 50)) $ITER_TOKENS)
LR_WSD_DECAY_ITERS=$(divide_rounding_up $((TRAIN_TOKENS / 5)) $ITER_TOKENS)
LR_DECAY_ITERS=$TRAIN_ITERS

: "${CHECKPOINT_PATH:=checkpoints/test}"
: "${DATA_CACHE_PATH:=../data_cache}"
: "${DATA_PATH:=/shared_silo/scratch/datasets/CC-MAIN-2020-high-actual/gpt-neox-20b/merged/CC-MAIN-2020-high-actual_merged}"
: "${TENSORBOARD_DIR:=./tensorboard}"
: "${WANDB_PROJECT:=}"
: "${WANDB_ENTITY:=}"
: "${WANDB_EXP_NAME:=}"
: "${WANDB_SAVE_DIR:=./wandb}"

MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 16
    --num-query-groups 4
    --group-query-attention
    --kv-channels 256
    --attention-backend auto
    --attention-softmax-in-fp32
    --position-embedding-type rope
    --rotary-base 10000000
    --rotary-percent 0.25
    --qk-layernorm
    --attention-output-gate
    --apply-wd-to-qk-layernorm
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
    # Hybrid attention (HF layer_types: 3x linear_attention + 1x full_attention)
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq 4
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 16
    --linear-num-value-heads 32
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --no-masked-softmax-fusion
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-create-attention-mask-in-dataloader
    --sequence-parallel
    --no-rope-fusion
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE:-1}
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --use-distributed-optimizer
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr $LR
    --min-lr $MIN_LR
    --train-iters $TRAIN_ITERS
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr-decay-style "WSD"
    --lr-wsd-decay-style "linear"
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --clip-grad 1.0
    --weight-decay 0.05
    --log-throughput
    --log-timers-to-tensorboard
    --log-progress
    --log-interval 1
    --eval-interval 100
    --eval-iters 10
    --ckpt-format torch
    --auto-detect-ckpt-format
    --make-vocab-size-divisible-by 256
    --dataloader-type single
    --num-workers 2
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --num-workers 7
    --num-dataset-builder-threads 7
    --split '99,1,0'
    # --tokenizer-model Qwen/Qwen3.5-9B
    --tokenizer-model EleutherAI/gpt-neox-20b
    --tokenizer-type HuggingFaceTokenizer
    --max-position-embeddings $SEQ_LENGTH
    --seq-length $SEQ_LENGTH
)

OUTPUT_ARGS=(
    --save-interval 10000
    --ckpt-format torch_dist
    --log-throughput
    --log-timers-to-tensorboard
    --log-progress
    --log-interval 1
    --eval-interval 10000
    --eval-iters 100
    --wandb-project $WANDB_PROJECT
    --wandb-entity $WANDB_ENTITY
    --wandb-exp-name $WANDB_EXP_NAME
    --wandb-save-dir $WANDB_SAVE_DIR
    --tensorboard-dir $TENSORBOARD_DIR
)
