MIN_LR=${MIN_LR:-0}
LR=${LR:-3e-4}
WARMUP_FRACTION=${WARMUP_FRACTION:-1/10}
COOLDOWN_FRACTION=${COOLDOWN_FRACTION:-1/5}
# Calculate TRAIN_ITERS from TRAIN_TOKENS
TRAIN_TOKENS=30_000_000_000   # TRAIN_ITERS computed from this
TRAIN_TOKENS=${TRAIN_TOKENS//_}    # drop "_" for bash math
ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))

divide_rounding_up() {
    echo $((($1+$2-1)/$2))
}

TRAIN_ITERS=$(divide_rounding_up $TRAIN_TOKENS $ITER_TOKENS)

WARMUP_CALC=$((TRAIN_ITERS * ${WARMUP_FRACTION}))
LR_WARMUP_ITERS=$WARMUP_CALC
LR_WSD_DECAY_ITERS=$((TRAIN_ITERS*${COOLDOWN_FRACTION}))
LR_DECAY_ITERS=$TRAIN_ITERS

    # --no-create-attention-mask-in-dataloader
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

TRAINING_ARGS=(
    --global-batch-size $GLOBAL_BATCH_SIZE
    --no-async-tensor-model-parallel-allreduce
    --no-masked-softmax-fusion
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-rope-fusion
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    # Muon optimizer does not support distributed optimizer
    # --use-distributed-optimizer
    # Muon optimizer does not support overlap grad reduce/param gather
    # --overlap-grad-reduce
    # --overlap-param-gather
    --optimizer muon
    --muon-momentum 0.9
    --muon-scale-mode spectral
    --muon-extra-scale-factor 0.2
    --muon-num-ns-steps 5
    --muon-tp-mode blockwise
    --muon-fp32-matmul-prec medium
    # Adam config for non-linear parameters (Muon handles linear params)
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-8
    --lr $LR
    --min-lr $MIN_LR
    --train-iters $TRAIN_ITERS
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr-decay-style "WSD"
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --lr-wsd-decay-style "linear"
    --clip-grad 1.0
    --weight-decay 0.05
    --eval-interval 10000
    --eval-iters 100
    --make-vocab-size-divisible-by 256
    --dataloader-type single
    --num-workers 7
)

OUTPUT_ARGS=(
    --save $CHECKPOINT_PATH
    --save-interval 10000
    # Muon optimizer supports torch and torch_dist checkpoint formats
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