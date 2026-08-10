# coding=utf-8
# Copyright 2024 Guyu AI and the HuggingFace Inc. team. All rights reserved.
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

"""Vocab32kGPT2Moe model configuration."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Vocab32kGPT2MoeConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Vocab32kGPT2MoeModel`]. It is used to instantiate
    a GPT-2 style decoder in which part of the feed-forward blocks are replaced by a top-k routed mixture of experts.
    Instantiating a configuration with the defaults will yield a GPT-2 base backbone (12 layers, 768 hidden, 12 heads)
    paired with a 32K SentencePiece vocabulary and dense feed-forward blocks.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Vocab32kGPT2MoeModel`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`str`, *optional*, defaults to `"cls_index"`):
            Sequence-summary strategy inherited from the GPT-2 configuration. Unused by the causal LM heads shipped in
            this repository, kept so that GPT-2 checkpoints remain loadable. One of `"last"`, `"first"`, `"mean"`,
            `"cls_index"` or `"attn"`.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Whether or not to add a projection after the sequence-summary vector extraction.
        summary_activation (`str`, *optional*):
            Activation applied on top of the sequence-summary projection. Pass `"tanh"` for a tanh activation, any
            other value results in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Whether the sequence-summary projection outputs should have `config.num_labels` or `config.hidden_size`
            classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio applied after the sequence-summary projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by `sqrt(head_dim)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        bos_token_id (`int`, *optional*, defaults to 1):
            Id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 2):
            Id of the end of sentence token in the vocabulary.
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.
        n_shared_experts (`int`, *optional*):
            Number of always-active shared experts evaluated in parallel with the routed experts and added to their
            output. `None` disables shared experts. Requires `moe_intermediate_size` to be set, because the shared
            expert width is `moe_intermediate_size * n_shared_experts`.
        n_routed_experts (`int`, *optional*):
            Number of routed experts per MoE block. `None` keeps every feed-forward block dense, which reduces the
            model to plain GPT-2.
        num_experts_per_tok (`int`, *optional*):
            Number of experts each token is dispatched to (the `k` of top-k routing).
        moe_layer_freq (`int`, *optional*, defaults to 1):
            Convert every `moe_layer_freq`-th decoder layer into an MoE layer. `1` means every layer.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Keep the first `first_k_dense_replace` decoder layers dense before starting to insert MoE blocks.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to renormalize the top-k routing weights so that they sum to one.
        scoring_func (`str`, *optional*, defaults to `"softmax"`):
            Function used to turn router logits into expert affinities. Only `"softmax"` is implemented.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Weight of the load-balancing auxiliary loss. `0.0` disables it.
        seq_aux (`bool`, *optional*, defaults to `True`):
            Whether to compute the auxiliary load-balancing loss per sequence instead of over the whole batch.

    Example:

    ```python
    >>> from models import Vocab32kGPT2MoeConfig, Vocab32kGPT2MoeForCausalLM

    >>> # A 12-layer GPT-2 backbone with 8 experts per layer and top-2 routing
    >>> configuration = Vocab32kGPT2MoeConfig(n_routed_experts=8, num_experts_per_tok=2)

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Vocab32kGPT2MoeForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vocab_32k_gpt2moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=32000,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        n_shared_experts=None,
        n_routed_experts=None,
        num_experts_per_tok=None,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        scoring_func="softmax",
        aux_loss_alpha=0.001,
        seq_aux=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


# Deprecated alias kept for configs and scripts written against the original class name.
vocab_32k_gpt2moeConfig = Vocab32kGPT2MoeConfig