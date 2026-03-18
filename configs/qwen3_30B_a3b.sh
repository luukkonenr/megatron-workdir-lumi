MIN_LR=${MIN_LR:-0}
LR=${LR:-3e-4}

# If TRAIN_ITERS is set externally, use it; otherwise default to a small number for testing
TRAIN_ITERS=${TRAIN_ITERS:-500}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-$TRAIN_ITERS}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-100}
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS:-$((LR_DECAY_ITERS / 2))}

MODEL_ARGS=(
    --num-layers 48
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 32
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
    --init-method-std 0.006
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --bf16
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 151936
    --num-experts 128
    --moe-router-topk 8
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type aux_loss
    --moe-router-force-load-balancing   
    --moe-aux-loss-coeff 0.001
    --moe-z-loss-coeff 1e-4
    --moe-router-dtype fp32
    --expert-model-parallel-size 16
    --moe-token-dispatcher-type alltoall
    --moe-token-dispatcher-type flex
    # --moe-flex-dispatcher-backend hybridep
    --moe-flex-dispatcher-backend deepep 
    --moe-enable-deepep # This doesn't work on the current container
    # --moe-hybridep-num-sms 32
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-fusion
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size ${GLOBAL_BATCH_SIZE:-128}
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
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
    # --no-async-tensor-model-parallel-allreduce
    # --no-masked-softmax-fusion
    # --no-gradient-accumulation-fusion
    # --no-bias-dropout-fusion
    # --no-rope-fusion
    --overlap-grad-reduce
    --overlap-param-gather
    --overlap-moe-expert-parallel-comm
    --recompute-granularity selective
    --recompute-modules core_attn moe_act layernorm
    # --delay-wgrad-compute
    --cross-entropy-loss-fusion
    # --fp8-format hybrid
    # --fp8-recipe tensorwise
    # --fp8-amax-compute-algo max
    # --fp8-amax-history-len 16
    # --use-precision-aware-optimizer
    # --main-grads-dtype bf16
    # --exp-avg-dtype fp8
    # --exp-avg-sq-dtype fp8
)

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
    --save $CHECKPOINT_PATH
    --save-interval 1000
    --ckpt-format torch_dist
    --auto-detect-ckpt-format
    --log-throughput
    --log-timers-to-tensorboard
    --log-progress
    --log-interval 1
    --eval-interval 100
    --eval-iters 10
)
