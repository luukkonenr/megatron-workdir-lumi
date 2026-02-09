MODEL_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model meta-llama/Llama-3.1-8B-Instruct
    --num-layers 32 
    --hidden-size 4096 
    --ffn-hidden-size 14336 
    --num-attention-heads 32 
    --num-query-groups 8
    --group-query-attention 
    --use-flash-attn 
    --attention-softmax-in-fp32 
    --max-position-embeddings 8192
    --seq-length 8192
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
)

# Set small defaults to avoid errors when not using them
: "${TRAIN_ITERS:=20}"
: "${LR_DECAY_ITERS:=20}"
: "${LR_WSD_DECAY_ITERS:=10}"
: "${CHECKPOINT_PATH:=checkpoints/test}"
: "${DATA_CACHE_PATH:=./data_cache}"

TRAINING_ARGS=(
    --micro-batch-size 2 
    --global-batch-size 2048 
    --no-async-tensor-model-parallel-allreduce 
    --no-masked-softmax-fusion 
    --no-gradient-accumulation-fusion 
    --no-bias-dropout-fusion 
    --no-rope-fusion 
    --distributed-timeout-minutes 60 
    --tensor-model-parallel-size 1 
    --pipeline-model-parallel-size 1 
    --context-parallel-size 1 
    --use-distributed-optimizer 
    # --data-path $DATA_PATH 
    --mock-data
    --optimizer adam 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --adam-eps 1e-8 
    --lr 3e-4 
    --min-lr 0.0 
    --train-iters $TRAIN_ITERS 
    --lr-warmup-iters 25000 
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
    --data-cache-path $DATA_CACHE_PATH 
    --make-vocab-size-divisible-by 256 
    --dataloader-type single 
    --num-workers 2 
)