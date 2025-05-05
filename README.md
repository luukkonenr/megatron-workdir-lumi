# megatron-workdir-lumi

Workdir mainly for internal usage. Requires internal project credentials for direct usage. With project *353-access you can get a succesfull training started in seconds. 

## Training

This example has `--exit-interval 20` so it will terminate after 20 iterations. It can be adapted to your needs. 
Example uses Llama3.1-tokenizer, which has a vocab size of approximately 128k tokens.

This training setup scales almost linearly up to 64 nodes with TP=2. 

| TGS | NODES |
|------|----------|
| 1403.29749  | 16 |
| 1381.689321  | 32 |
| 1314.907184  | 64 | 

```
git clone --recurse-submodules https://github.com/luukkonenr/megatron-workdir-lumi.git
cd megatron-workdir-lumi/
patch -p1 -i patches/checkpoint_conversion.diff

sbatch train-sbatch.sh TP=2 MODEL_SIZE=7B
```

Inspecting logs:
`tail -f logs/latest*`


## Tools 
### Conversion from huggingface
Megatron offers tools for conversion from Huggingface-format. This is an example of converting Llama3.1-8B to Megatron

* Note: Latest transformers don't work, plausibly due to this [refactor](https://github.com/huggingface/transformers/commit/071a161d3e38f56dbda2743b979f0afeed2cd4f1
) to`from_pretrained`-method, so you need to install e.g. `transformers==4.48.2`. Installation is set to run at the start of conversion in [convert_llama3.1-8B.sh](convert_llama3.1-8B.sh)


#### Run conversion:

```sbatch scripts/convert_llama3.1-8B_hf_to_meg.sh```
There are some error messages after the conversion has run, but won't affect the saved checkpoints.

You can start continued pre-training with newly converted checkpoints with

```
sbatch train-sbatch.sh TP=2 MODEL_SIZE=7B LOAD_CKPT_PATH="megatron-checkpoints/llama3.1-8B-TP-2-PP-1" SAVE_CKPT_PATH="megatron-checkpoints/llama3.1-8B-TP-2-PP-1"
```

### Conversion to huggingface
1) Copy tools/saver_llama_mistral.py to Megatron-LM/tools/checkpoint/ `cp tools/saver_llama_mistral.py Megatron-LM/tools/checkpoint/`
2) Run `sbatch scripts/convert_llama3.1-8B_meg_to_hf.sh <path_to_checkpoint>`

This should work for other model sizes too and for both model types, with tied or untied output embeddings.


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




## Known Issues: 
*  Currently there seems to be a bug in
https://github.com/ROCm/Megatron-LM/blob/99bb7a92291528fe713618b355b1b9b31d3b3b9f/megatron/training/arguments.py#L709
Change that line in megatron/training/arguments.py from 
`if args.tensor_model_parallel_size > 1` to `if args.tensor_model_parallel_size > 1 and args.num_experts:` to get conversion working.
* Conversion from HF -> Meg succeeds but there is some error message that pops up after the conversion is run, causing the finalize-step to crash the run. Checkpoint seems to be still okay. 

