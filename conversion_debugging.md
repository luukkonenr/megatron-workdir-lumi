# Debugging the differences between Megatron-LM and HF model results

Making conversion scripts between checkpoints of different frameworks is error-prone and often a bit tedious. As each of the functions are required to match, different implementations for example on query-key-layernorm might results in incompatibilities. 

Simplest approach for identifying where possinble differences is straightforward: simply compare outputs after each function and see that they are equivalent. This requires 
1) Running inference with framework A (here Megatron-LM) on some input.
2) Running inference with framework B (here transformers) on the same input.
3) Adding print-statements after each function call to compare hidden states.

To do this with transformers, install an editable version with
```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

and get to work.
