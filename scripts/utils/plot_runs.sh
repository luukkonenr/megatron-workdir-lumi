SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd 
CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0.sif
# Add all runs you want into single script
singularity exec -B $SINGULARITY_BIND $CONTAINER \
    python3 tools/plot_multiple_runs.py \
        --logs \
            /scratch/project_462000353/avirtanen/Megatron-LM-rocm/logs/meglm-qwen3-train-13787989-2025-10-21_01-02-32.out \
            /scratch/project_462000353/avirtanen/Megatron-LM-rocm/logs/meglm-dense-30B-train-13812501-2025-10-21_19-02-48.out \
            /scratch/project_462000353/avirtanen/Megatron-LM-rocm/logs/meglm-dense-8B-train-13791260-2025-10-21_03-48-50.out \
            /scratch/project_462000353/avirtanen/Megatron-LM-rocm/logs/meglm-dense-3B-train-13813144-2025-10-21_19-02-51.out \
        --out-dir plots --plot-by iters --keys 'lm loss' 'learning_rate'
