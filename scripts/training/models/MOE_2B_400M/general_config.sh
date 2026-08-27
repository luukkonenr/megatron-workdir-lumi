TRAIN_TOKENS=30_000_000_000

# Calculate training samples based on tokens
TRAIN_TOKENS=$(echo $TRAIN_TOKENS | sed 's/_//g')
SEQ_LENGTH=2048
TRAIN_SAMPLES=$(($TRAIN_TOKENS / $SEQ_LENGTH))

DATA=(
    /pfs/lustrep4/scratch/project_462000963/preprocessed/gpt-neox-20b/nemotron-cc/1.0/high-actual
    /pfs/lustrep4/scratch/project_462000963/preprocessed/gpt-neox-20b/nemotron-cc/1.0/high-synthetic-distill
    /pfs/lustrep4/scratch/project_462000963/preprocessed/gpt-neox-20b/nemotron-cc/1.0/high-synthetic-diverse_qa_pairs
    /pfs/lustrep4/scratch/project_462000963/preprocessed/gpt-neox-20b/nemotron-cc/1.0/high-synthetic-extract_knowledge
    /pfs/lustrep4/scratch/project_462000963/preprocessed/gpt-neox-20b/nemotron-cc/1.0/high-synthetic-knowledge_list
    /pfs/lustrep4/scratch/project_462000963/preprocessed/gpt-neox-20b/nemotron-cc/1.0/high-synthetic-wrap_medium
)

MOE_FFN_HIDDEN_SIZE=768

MODEL_ARGS=(
    # General architecture
    --num-layers 12
    --hidden-size 1024
    --ffn-hidden-size 3584
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE
    --num-attention-heads 16
    --qk-layernorm
    --kv-channels 64
    # MOE specific args
    --num-experts $NUM_EXPERTS
    --moe-router-topk $MOE_TOPK
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-grouped-gemm
    --moe-token-dispatcher-type allgather 
    --bf16
    --disable-bias-linear
    --max-position-embeddings 2048
    --norm-epsilon 1e-6
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --no-position-embedding
    --rotary-base 1000000
    --rotary-percent 1.0
    # --moe-router-dtype fp32
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-create-attention-mask-in-dataloader
)
    # --untie-embeddings-and-output-weights
# if num shared experts > 0, add moe-shared-expert-intermediate-size to MODEL_ARGS. MOE_NUM_SHARED_EXPERTS must be set in the calling script.
if [ "$MOE_NUM_SHARED_EXPERTS" -gt 0 ]; then
    MODEL_ARGS+=(
        --moe-shared-expert-intermediate-size $((MOE_FFN_HIDDEN_SIZE * $MOE_NUM_SHARED_EXPERTS))
    )
fi
DATA_ARGS=(
    --seq-length $SEQ_LENGTH
    --data-path ${DATA[@]}
    --data-cache-path "/flash/project_462000353/rluukkon/cache"
    --split 99,1,0
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model EleutherAI/gpt-neox-20b
)
INFRA_ARGS=(
    --use-distributed-optimizer
    --overlap-param-gather
    --overlap-grad-reduce
    --distributed-timeout-minutes 30
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --use-flash-attn
    --attention-softmax-in-fp32
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-bias-swiglu-fusion
    --num-workers 7
)
    # --no-masked-softmax-fusion

TRAIN_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size 1024
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style WSD
    --lr-warmup-fraction 0.01
    --lr-wsd-decay-samples $((TRAIN_SAMPLES / 5))
    --train-samples $TRAIN_SAMPLES
)

SAVE_ARGS=(
    --log-interval $LOG_INTERVAL
    --log-throughput
    --save $SAVE_DIR
    --load $LOAD_DIR
    --save-interval $SAVE_INTERVAL
    --eval-interval $EVAL_INTERVAL
    --wandb-save-dir $SAVE_DIR/wandb
    --wandb-project moe-exploration
    --wandb-exp-name ${MODEL_NAME}_${SLURM_JOB_ID}
    --tensorboard-dir $SAVE_DIR/tensorboard
)    

    # --moe-router-force-load-balancing
    # --moe-permute-fusion # TE version conflict



