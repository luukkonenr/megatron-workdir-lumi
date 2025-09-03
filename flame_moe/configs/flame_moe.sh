#!/bin/bash
# Model configuration with FLAME MoE.

MODEL_ARGS=(
    # Network Size
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-layers $NUM_LAYERS
    --num-attention-heads 16
    --swiglu
    --max-position-embeddings 2048
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --disable-bias-linear

    # Mixture of Experts
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE
    --num-experts 64
    --moe-router-topk 6
    --enable-shared-expert
    --moe-shared-expert-intermediate-size $((2 * MOE_FFN_HIDDEN_SIZE))
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-grouped-gemm
    # --moe-router-dtype fp32 # Not available on ROCM/Megatron-LM
    --moe-router-pre-softmax
    --moe-router-score-function softmax
    --moe-aux-loss-coeff 0.01
    --moe-z-loss-coeff 0.001

    # Regularization
    --hidden-dropout 0.0
    --attention-dropout 0.0

    # Initialization
    --init-method-std 0.02

    # Tokenizer
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model EleutherAI/pythia-12b
)


INFRA_ARGS=(
    --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
    --expert-model-parallel-size $EXPERT_MODEL_PARALLEL_SIZE
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --moe-token-dispatcher-type alltoall
    --distributed-timeout-minutes 30
    --bf16
)

TRAIN_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size 1024
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style WSD
    --lr-warmup-fraction 0.01
    --lr-wsd-decay-iters $((TRAIN_ITERS / 10))
    --train-iters $TRAIN_ITERS
)

DATA_ARGS=(
    --seq-length 2048
    --data-path flame_moe/data/merged_tokenized
    --split 90,5,5
)

SAVE_ARGS=(
    --log-interval 5
    --log-throughput
    # --save $SSD_WEIGHTS
    # --save-interval $SAVE_INTERVAL
    # --load $SSD_WEIGHTS
    --eval-interval $EVAL_INTERVAL
    # --wandb-save-dir $SSD_WEIGHTS
    # --wandb-project $WANDB_PROJECT
    # --wandb-exp-name $SLURM_JOB_ID
    # --tensorboard-dir $SSD_WEIGHTS
)