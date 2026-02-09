#!/bin/bash
for GLOBAL_BATCH_SIZE in 1024 4096; do
    for MOE_ROUTER_TOPK in 4 8; do
        for LR in 3e-3 5e-3; do
            for NUM_EXPERTS in 64; do
                LR=${LR} NUM_EXPERTS=${NUM_EXPERTS} MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} \
                sbatch scripts/training/oellm_sweeps/train-oellm-model.sh
                exit 0
            done
        done
    done
done
