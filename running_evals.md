# Running evals with Megatron-LM checkpoints and eval harness:
Clone the following forks and branches:

Megatron: https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM/tree/dev
  - Contains a fix for reading --qk-layernorm and --norm-epsilon from the checkpoint args. If running a model with qk-layernorm without this patch, lm_eval requires extra_args="--qk-layernorm" or it ends up building a broken model.

eval-harness: https://github.com/luukkonenr/lm-evaluation-harness/tree/upstream
  - Contains a patch to do gather-calls from lm-object directly, as the old API used `lm.accelerator.gather_object`


For example:
```
mkdir evaluations
cd evaluations
git clone -b dev https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM/
git clone -b upstream https://github.com/luukkonenr/lm-evaluation-harness
cd lm-evaluation-harness

```

## Example running with both Megatron-LM and HF checkpoint formats

### 1 node with 8 GPUS

Note: this needs to be executed on a GPU node inside the execution environment. 
For example. 
`srun_wrapper.sh`
```
GPUS_PER_NODE=8
TASKS_PER_NODE=1
TOTAL_CPUS=100
CPUS_PER_TASK=$((TOTAL_CPUS / TASKS_PER_NODE))
srun --time 0-01:00:00 \
    --ntasks-per-node=$TASKS_PER_NODE \
    --gpus-per-node=$GPUS_PER_NODE \
    --cpus-per-task=$CPUS_PER_TASK \
    --nodes=1 \
    --mem=0 \
    --partition=$PARTITION \
    --account=$ACCOUNT \
    --pty bash -c " \
        export MASTER_ADDR=$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
        export MASTER_PORT=9999
        export WORLD_SIZE=\$SLURM_NTASKS
        apptainer exec $CONTAINER bash $@"
```
`run_evals.sh`
```
MEGATRON_PATH=PATH/TO/Megatron-LM
LM_EVAL_PATH=PATH/TO/lm-evaluation-harness
export PYTHONPATH=$PYTHONPATH:$MEGATRON_PATH:$LM_EVAL_PATH

CHECKPOINT=<>
TOKENIZER=<>
For example:
# CHECKPOINT="/shared_silo/scratch/rluukkon/megatron-workdir-lumi/checkpoints/qwen3-30b-a3b-bridge-test"
# TOKENIZER="Qwen/Qwen3-30B-A3B"

result_dir="results"
# Megatron
num_fewshot=0
devices=8
batch_size=32
tasks="arc_easy"
output_name="qwen3-30b-a3b-arc_easy-num_fewshot-$num_fewshot.json"

torchrun --nproc_per_node=$devices --master_port=9999 $LM_EVAL_PATH/lm_eval/__main__.py \
    --model megatron_lm \
    --model_args load=$CHECKPOINT,tokenizer_type=HuggingFaceTokenizer,tokenizer_model=$TOKENIZER,devices=$devices,micro_batch_size=$batch_size \
    --tasks $tasks \
    --num_fewshot $num_fewshot \
    --batch_size $batch_size \
    --log_samples \
    --output_path $result_dir/megatron_$output_name
    

# HF 
accelerate launch -m lm_eval --model hf \
    --tasks $tasks \
    --log_samples \
    --model_args pretrained=$TOKENIZER,dtype=bfloat16 \
    --batch_size auto \
    --num_fewshot $num_fewshot \
    --output_path $result_dir/hf_$output_name
```
