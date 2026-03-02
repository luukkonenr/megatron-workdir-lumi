#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --account=project_462000963
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/convert-%j.out
#SBATCH --error=logs/convert-%j.err

# Convert open-sci model with ~0.05B non-embedding parameters from
# Megatron to HF format.

# ln -sf "convert-${SLURM_JOBID}.out" "logs/latest.out"
# ln -sf "convert-${SLURM_JOBID}.err" "logs/latest.err"

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 CHECKPOINT TOKENIZER OUTDIR"
    exit 1
fi

CHECKPOINT="$1"
TOKENIZER="$2"
OUTDIR="$3"

# This script hard-codes the model architecture and at least the
# settings below will need to be updated if the architecture is
# changed.
hidden_size=384
intermediate_size=1536
max_position_embeddings=4096
num_key_value_heads=6 # need to pass separately as stored as 1 in checkpoint
num_attention_heads=6
num_layers=12

# This script uses the conversion from open-sci with a few tweaks
# for LUMI. You can get this with
#     git clone https://github.com/OpenEuroLLM/Megatron-LM-Open-Sci.git \
#         -b converter
# and then adjust the following path to point there.
MEGATRON_OPEN_SCI_PATH="Megatron-LM-Open-Sci"

# Path to Megatron (required by conversion script)
MEGATRON_PATH="/scratch/project_462000963/users/pyysalos/ROCm-Megatron-LM"
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH:-}

module use /appl/local/csc/modulefiles; module load pytorch

export CC=gcc-12
export CXX=g++-12

mkdir -p "$OUTDIR"

# Get vocab size from tokenizer
vocab_size=$(python3 -c "
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('$TOKENIZER')
print(t.vocab_size)")

# Write config.json
cat <<EOF > ${OUTDIR}/config.json
{
    "_name_or_path": "",
    "architectures": [
      "OpensciForCausalLM"
    ],
    "attention_bias": true,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_opensci.OpensciConfig",
        "AutoModel": "modeling_opensci.OpensciPreTrainedModel",
        "AutoModelForCausalLM": "modeling_opensci.OpensciForCausalLM"
      },
    "bos_token_id": 0,
    "eos_token_id": 0,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": $hidden_size,
    "initializer_range": 0.02,
    "intermediate_size": $intermediate_size,
    "max_position_embeddings": $max_position_embeddings,
    "model_type": "opensci",
    "num_attention_heads": $num_attention_heads,
    "num_hidden_layers": $num_layers,
    "num_key_value_heads": $num_key_value_heads,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 10000,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.48.3",
    "use_cache": true,
    "vocab_size": $vocab_size
  }
EOF

# Download modeling_opensci.py and configuration_opensci.py
cp /pfs/lustrep4/scratch/project_462000963/users/rluukkon/git/transformers/src/transformers/models/opensci/modeling_opensci.py  "$OUTDIR/modeling_opensci.py"
cp /pfs/lustrep4/scratch/project_462000963/users/rluukkon/git/transformers/src/transformers/models/opensci/configuration_opensci.py  "$OUTDIR/configuration_opensci.py"
# replace relative paths in the files with absolute paths. 
# These are in the form of "from ...<some>.<path>" --> "from transformers.<some>.<path>" 
# for example from ...activations import ACT2FN

sed -i \
  -e 's|from \.\.\.\([a-zA-Z_][a-zA-Z0-9_.]*\) import|from transformers.\1 import|g' \
  "$OUTDIR/modeling_opensci.py"

  # -e 's|from \.\([a-zA-Z_][a-zA-Z0-9_.]*\) import|from transformers.models.opensci.\1 import|g' \
sed -i \
  -e 's|from \.\.\.\([a-zA-Z_][a-zA-Z0-9_.]*\) import|from transformers.\1 import|g' \
  "$OUTDIR/configuration_opensci.py"


# Run conversion script
python -u "$MEGATRON_OPEN_SCI_PATH/scripts/ckpt/mcore_to_hf_opensci.py" \
       --load_path "$CHECKPOINT" \
       --save_path "$OUTDIR" \
       --source_model "$OUTDIR" \
       --target_tensor_model_parallel_size 1 \
       --target_pipeline_model_parallel_size 1 \
       --world_size 1 \
       --target_params_dtype "bf16" \
       --convert_checkpoint_from_megatron_to_transformers \
       --num_key_value_heads $num_key_value_heads \
       --print-checkpoint-structure

# Save tokenizer
echo "Saving tokenizer"
python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$TOKENIZER')
t.save_pretrained('${OUTDIR}')"

# Test the converted model
echo "Testing generation"
prompt="The best advice I ever heard is this:"
python3 -c "
from transformers import pipeline, AutoTokenizer
pipe = pipeline(
    'text-generation',
    model='$OUTDIR',
    tokenizer=AutoTokenizer.from_pretrained('$OUTDIR'),
    device='cuda', trust_remote_code=True
)
print(pipe('$prompt')[0]['generated_text'])
"
