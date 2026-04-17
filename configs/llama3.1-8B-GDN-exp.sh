MIN_LR=${MIN_LR:-0}
LR=${LR:-3e-4}
# Calculate TRAIN_ITERS from TRAIN_TOKENS
TRAIN_TOKENS=30_000_000_000   # TRAIN_ITERS computed from this
TRAIN_TOKENS=${TRAIN_TOKENS//_}    # drop "_" for bash math
ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))

divide_rounding_up() {
    echo $((($1+$2-1)/$2))
}

TRAIN_ITERS=$(divide_rounding_up $TRAIN_TOKENS $ITER_TOKENS)

LR_WARMUP_ITERS=$(divide_rounding_up $((TRAIN_TOKENS / 50)) $ITER_TOKENS)  # ~2% warmup
LR_WSD_DECAY_ITERS=$(divide_rounding_up $((TRAIN_TOKENS / 5)) $ITER_TOKENS)  # 20% cooldown
LR_DECAY_ITERS=$TRAIN_ITERS


MODEL_ARGS=(
    --num-layers 32 
    --hidden-size 4096 
    --ffn-hidden-size 14336 
    --num-attention-heads 32 
    --num-query-groups 8
    --group-query-attention 
    --attention-backend auto
    --attention-softmax-in-fp32 
    --position-embedding-type rope 
    --rotary-base 500000
    --rotary-percent 1.0
    --rotary-scaling-factor 8.0
    --disable-bias-linear 
    --init-method-std 0.02 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --normalization RMSNorm 
    --bf16 
    --swiglu 
    --untie-embeddings-and-output-weights
    # Gated Delta Net (GDN) configuration
    # Replace 3/4 of attention layers with GDN: pattern [1,1,1,0] repeating
    # (1=GDN, 0=SDPA; every 4th layer keeps standard attention)
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq 4
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 8
    --linear-num-value-heads 32
    # Use PyTorch fallback for GDN kernel instead of flash-linear-attention (fla)
    # --deterministic-mode
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    # --no-async-tensor-model-parallel-allreduce 
    --no-masked-softmax-fusion 
    --no-gradient-accumulation-fusion 
    --no-bias-dropout-fusion 
    --no-create-attention-mask-in-dataloader
    --sequence-parallel
    --no-rope-fusion 
    --distributed-timeout-minutes 60 
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1 
    --context-parallel-size 1
    --use-distributed-optimizer 
    --optimizer adam 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --adam-eps 1e-8 
    --lr 3e-4 
    --min-lr 0.0 
    --train-iters $TRAIN_ITERS 
    --lr-warmup-iters $LR_WARMUP_ITERS 
    --lr-decay-style "WSD" 
    --lr-wsd-decay-style "linear" 
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
    --ckpt-format torch 
    --auto-detect-ckpt-format 
    --make-vocab-size-divisible-by 256 
    --dataloader-type single 
    --num-workers 2 
)

DATA_ARGS=(
    --data-path /shared_silo/scratch/datasets/CC-MAIN-2020-high-actual/gpt-neox-20b/merged/CC-MAIN-2020-high-actual_merged
    --data-cache-path ../data_cache
    --num-workers 7
    --num-dataset-builder-threads 7
    --split '99,1,0'
    --tokenizer-model EleutherAI/gpt-neox-20b
    --tokenizer-type HuggingFaceTokenizer
    --max-position-embeddings $SEQ_LENGTH   
    --seq-length $SEQ_LENGTH
)

OUTPUT_ARGS=(
    # --save $CHECKPOINT_PATH
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