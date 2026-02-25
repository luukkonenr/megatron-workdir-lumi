MODEL_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen3-30B-A3B
    --num-layers 48 
    --hidden-size 2048 
    --ffn-hidden-size 6144 
    --num-attention-heads 32 
    --num-query-groups 4
    --group-query-attention 
    --qk-layernorm
    --use-flash-attn 
    --attention-softmax-in-fp32 
    --max-position-embeddings 40960
    --seq-length 4096
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
    --num-experts 384
    --moe-router-topk 8
    --moe-router-num-groups 8
    # --moe-router-group-topk 4
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type global_aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-router-dtype fp32
    --expert-model-parallel-size 8
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
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
    --micro-batch-size 4
    --global-batch-size 512
    # --no-async-tensor-model-parallel-allreduce 
    --no-masked-softmax-fusion 
    --overlap-grad-reduce
    --overlap-param-gather
    --no-gradient-accumulation-fusion 
    --no-bias-dropout-fusion 
    # --no-rope-fusion 
    --distributed-timeout-minutes 60 
    --tensor-model-parallel-size 1 
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
    --num-workers 2 
)