NUM_EXPERTS=${NUM_EXPERTS:-64}
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-8}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.001}

MODEL_ARGS=(
    --num-layers 9
    --hidden-size 512
    --ffn-hidden-size 1536 # not used in MOE
    --num-attention-heads 16
    --num-query-groups 16
    --group-query-attention
    --qk-layernorm
    --use-flash-attn
    --attention-softmax-in-fp32
    --max-position-embeddings 8192
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --bf16
    --swiglu
    --disable-bias-linear
    --moe-ffn-hidden-size 352
    --kv-channels 32
    --num-experts $NUM_EXPERTS
    --moe-router-topk $MOE_ROUTER_TOPK
    --moe-router-load-balancing-type aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff $MOE_AUX_LOSS_COEFF
    --moe-router-dtype fp32
    --expert-model-parallel-size 1
    --micro-batch-size 32
)