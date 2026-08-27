#!/bin/bash
#SBATCH --job-name=eval-harness
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --partition=dev-g
#SBATCH --time=00:90:00
#SBATCH --account=project_462000963
#SBATCH --exclusive
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err


ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.out logs/latest.out
ln -sf ${SLURM_JOB_NAME}-${SLURM_JOBID}.err logs/latest.err
export PWD=(`pwd -P`)
workdir=${PWD}
export PYTHONUSERBASE=".local"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS #This is valid only if ntasks==ngpus
export CUDA_DEVICE_MAX_CONNECTIONS=1 #This is needed for sequence paralellism
export CC=gcc-12
export CXX=g++-12
# SINGULARITY 
# sCONTAINER=/pfs/lustrep2/scratch/project_462000353/risto/containers/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-tev.2.2.0dev.sif
CONTAINER="/scratch/project_462000963/users/rluukkon/container_cache/MegatronTrainingLumi_x86_64.sif"
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd
# CHECKPOINT_PATH=checkpoints/flame-moe-419m-12872205/
# CHECKPOINT_PATH=c"checkpoints/flame-moe-290m-12969781/"
        # --tokenizer-model $TOKENIZER_MODEL


# CHECKPOINT_PATH="/scratch/project_462000963/users/pyysalos/experiments/500m-config/output/checkpoints"
CHECKPOINT_PATH="/pfs/lustrep4/scratch/project_462000963/users/rluukkon/output/moe-exploration_20251223_143858_670"
TOKENIZER_MODEL="EleutherAI/gpt-neox-20b"
# TOKENIZER_MODEL="open-ai/gpt-oss"
# RANDOM_DIR="/tmp/lm_eval_$(date +%s%N)"
timestamp=$(date +%s)
OUTPUT_DIR="eval_results/"
OUTPUT_FILE="${OUTPUT_DIR}/run_${timestamp}"
# mkdir -p "$RANDOM_DIR"
# echo Saving temporary results to $RANDOM_DIR
mkdir -p $OUTPUT_DIR
echo Final results will be saved to: $OUTPUT_FILE

# Adding lm-evaluation-harness to PYTHONPATH without installing it for dev purposes
export PYTHONPATH=$PYTHONPATH:lm-evaluation-harness
export PYTHONPATH=$PYTHONPATH:NVIDIA-Megatron-LM
        # --use-mp-args-from-checkpoint-args # use model parallel args from checkpoint
megatron_arguments=(
        --load $CHECKPOINT_PATH
        --use-checkpoint-args # use model args from checkpoint
        --no-load-optim 
        --no-load-rng 
        --max-tokens-to-oom 40000
        --micro-batch-size 2
        --use-legacy-static-engine
        --bf16
        --use-flash-attn
        --qk-layernorm
        --use-legacy-static-engine
        --dist-ckpt-strictness log_unexpected
        --tokenizer-type HuggingFaceTokenizer
        )
        # --rotary-base 500000



# TASKS="arc_easy,arc_challenge,piqa,hellaswag,openbookqa,mmlu,lambada_openai,winogrande,boolq,commonsense_qa"
srun --label \
    singularity exec \
    -B ${PWD} \
    $CONTAINER \
    ./launcher.sh \
    ${workdir}/NVIDIA-Megatron-LM/tools/run_text_generation_server.py ${megatron_arguments[@]}
    
#     # --output_path $RANDOM_DIR \
# python lm_eval --model_args "pretrained=$MODEL,trust_remote_code=True" --device cuda:0 --batch_size 32 --tasks "$TASKS" --num_fewshot 0 --output_path results    
 
# echo Moving temporary results from $RANDOM_DIR to $OUTPUT_FILE
# find "$RANDOM_DIR" -name "results_*.json" -exec mv {} "$OUTPUT_FILE" \;
# rm -rf "$RANDOM_DIR"