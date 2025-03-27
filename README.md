# megatron-workdir-lumi

Workdir mainly for internal usage. Requires internal project credentials for direct usage. With project *353-access you can start a succesfull training started in seconds. 

## Training

This example has `--exit-interval 20` so it will terminate after 20 iterations. It will can be adapted to your needs. Training scales almost linearly on Llama3.1-8B up to 64 nodes with TP=2. 
Example uses Llama3.1-tokenizer, which is 128k tokens.
```
git clone --recurse-submodules https://github.com/luukkonenr/megatron-workdir-lumi.git
cd megatron-workdir-lumi/
sbatch train-sbatch.sh TP=2 MODEL_SIZE=7B
```

Inspecting logs:
`tail -f logs/latest*`


## Tools 
### Conversion from huggingface

Megatron offers direct conversion from Huggingface-format. These can be run interactively on a compute node like this: 
```
HF_FORMAT_DIR=/scratch/project_462000353/models/llama31-8b
TOKENIZER_MODEL=$HF_FORMAT_DIR
TARGET_PP=1
TARGET_TP=2
MEGATRON_FORMAT_DIR=megatron-checkpoints/llama3.1-8B-TP-$TARGET_TP-PP-$TARGET_PP
export CUDA_DEVICE_MAX_CONNECTIONS=1
python3 Megatron-LM/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama_mistral \
  --model-size llama3-8B \
  --checkpoint-type 'hf' \
  --saver mcore \
  --target-tensor-parallel-size ${TARGET_TP} \
  --target-pipeline-parallel-size ${TARGET_PP} \
  --load-dir ${HF_FORMAT_DIR} \
  --save-dir ${MEGATRON_FORMAT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL}
```

Or as slurm batch jobs e.g `sbatch convert_llama3.1-8B.sh`

## \# IMPORTANT NOTE
Currently there seems to be a bug in
https://github.com/ROCm/Megatron-LM/blob/99bb7a92291528fe713618b355b1b9b31d3b3b9f/megatron/training/arguments.py#L709
Change that line in megatron/training/arguments.py from 
`if args.tensor_model_parallel_size > 1` to `if args.tensor_model_parallel_size > 1 and args.num_experts:` to get conversion working.

### Conversion to huggingface
`TODO`


### Logfiles
#### Simple logfile summary
```
module use /appl/local/csc/modulefiles && module load pytorch
python3 tools/summarize.py --dir logs/ --filter="params=>7.9B" --short # or --csv for easy copy-paste
Found 2 files
           tgs     tflops  mem_usages seq_len m.b.s batch_size  world_size d.p.s   fsdp precision   fp8 optimizer   e.s t.m.p.s p.m.p.s   r.g n.m.p           timestamp  iters
0  1385.583976  80.242857      0.9783    8192     1        128          16     8  False  bfloat16  None      adam  8192       2       1  None  8.0B 2025-03-24 15:44:55      7
1  1385.583976  80.242857      0.9783    8192     1        128          16     8  False  bfloat16  None      adam  8192       2       1  None  8.0B 2025-03-24 15:44:55      7

```
#### Single file inspection
```
module use /appl/local/csc/modulefiles && module load pytorch
python3 tools/throughput.py logs/latest.out


File: logs/latest.out
seq_len: 8192, micro_batch_size: 1, batch_size: 128, world_size: 16, data_parallel_size: 8, fsdp: False, precision: torch.bfloat16, fp8: None, optimizer: adam, embedding_size: 8192, transformer_impl: transformer_engine, tensor_model_parallel_size: 2, pipeline_model_parallel_size: 1, recompute_granularity: None, num_model_params: 8.03B, timestamp: 2025-03-24 15:44:55, iters: 7, data_path: ['0.7', '/scratch/project_462000353/data/processed-llama31/merged/fi-culturax', '0.25', '/scratch/project_462000353/data/processed-llama31/merged/fineweb-edu-deduplicated', '0.04', '/scratch/project_462000353/data/processed-llama31/merged/starcoder', '0.01', '/scratch/project_462000353/data/processed-llama31/merged/xling'], 
Skipped first 2 iterations. Averaging over 7 iterations
-------------------------------------------------
           | TGS        | TFLOPs     | mem usages
-------------------------------------------------
mean       | 1385.58    | 80.24      | 0.98      
std        | 0.35       | 0.05       | 0.00      
max        | 1385.95    | 80.30      | 0.98      
min        | 1385.01    | 80.20      | 0.98
```

#### Param comparison of two logfiles

```
python3 tools/compare_params.py logs/testrun1.out logs/testrun2.out
num_layers                                     32  24
global_batch_size                              128  1024
tensor_model_parallel_size                     4  1
encoder_num_layers                             32  24
sequence_parallel                              True  False
num_query_groups                               8  1
hidden_size                                    4096  2048
expert_tensor_parallel_size                    4  1
group_query_attention                          True  False
ffn_hidden_size                                14336  8192
kv_channels                                    128  64
data_parallel_size                             4  16
moe_ffn_hidden_size                            14336  8192
```


