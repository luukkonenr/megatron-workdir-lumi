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
