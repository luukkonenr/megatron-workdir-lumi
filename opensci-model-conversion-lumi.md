Process to use modular transformers for custom models:


```
git clone https://github.com/huggingface/transformers.git
cd transformers
```
Install as an editable locally or use with appending to PYTHONPATH. Whatever works. Install libcst if your python env is missing it. 
```
mkdir src/transformers/models/opensci
```
Create modular-file under the created repo. This model is Qwen3 with MLP-bias read from config file.
```
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenSci model."""

from torch import nn

from ...utils import logging
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3ForQuestionAnswering,
    Qwen3ForSequenceClassification,
    Qwen3ForTokenClassification,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from ..qwen3.configuration_qwen3 import Qwen3Config


logger = logging.get_logger(__name__)


class OpensciConfig(Qwen3Config):
    r"""
    Configuration class for the OpenSci model. Identical to :class:`~transformers.Qwen3Config` except that the MLP
    layers use biases (``bias=True``).

    Refer to :class:`~transformers.Qwen3Config` for the full list of parameters.

    Args:
        mlp_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the MLP layers.

    Example::

        >>> from transformers import OpensciConfig, OpensciModel

        >>> configuration = OpensciConfig()
        >>> model = OpensciModel(configuration)
        >>> configuration = model.config
    """

    model_type = "opensci"

    def __init__(
        self,
        vocab_size: int | None = 151936,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 22016,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 32,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 32768,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_theta: float | None = 10000,
        rope_scaling: dict | None = None,
        attention_bias: bool | None = False,
        use_sliding_window: bool | None = False,
        sliding_window: int | None = 4096,
        max_window_layers: int | None = 28,
        layer_types: list[str] | None = None,
        attention_dropout: float | None = 0.0,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        mlp_bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            layer_types=layer_types,
            attention_dropout=attention_dropout,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.mlp_bias = mlp_bias


class OpensciRMSNorm(Qwen3RMSNorm):
    pass


class OpensciRotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class OpensciMLP(Qwen3MLP):
    def __init__(self, config):
        super().__init__(config)
        # Override projections to enable bias, which is the sole difference from Qwen3MLP.
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)


class OpensciAttention(Qwen3Attention):
    pass


class OpensciForCausalLM(Qwen3ForCausalLM):
    pass


class OpensciForSequenceClassification(Qwen3ForSequenceClassification):
    pass


class OpensciForTokenClassification(Qwen3ForTokenClassification):
    pass


class OpensciForQuestionAnswering(Qwen3ForQuestionAnswering):
    pass


__all__ = [
    "OpensciConfig",
    "OpensciForCausalLM",
    "OpensciForQuestionAnswering",
    "OpensciPreTrainedModel",  # noqa: F822
    "OpensciModel",  # noqa: F822
    "OpensciForSequenceClassification",
    "OpensciForTokenClassification",
]

```

After the modeling file is created, you can just use it with OpenSci-converter. Here's an example of how I use it. I have changed `q_layernorm` and `k_layernorm` from the [converter](github.com/LAION-AI/Megatron-LM-Open-Sci/blob/converter/scripts/ckpt/mcore_to_hf_opensci.py) to `q_norm` and `k_norm`
And one more thing: created modeling and configuration files are created to be part of the transformers model library, so to detach them and load with trust_remote_code=True, you need to replace relative imports with sed to use root "transformers." instead of "...<path>.<to>.<module>"


```
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

# Convert open-sci model with ~0.4B non-embedding parameters from
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
hidden_size=1024
intermediate_size=3840
max_position_embeddings=4096
num_key_value_heads=16 # need to pass separately as stored as 1 in checkpoint
num_attention_heads=16
num_layers=22

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
```
