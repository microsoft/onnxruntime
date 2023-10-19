import math
import torch
import transformers

import torch.nn.functional as F

from torch import nn
from typing import Optional, Tuple
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaModel,
    LlamaForCausalLM,
    rotate_half,
    repeat_kv,
    apply_rotary_pos_emb,
    _make_causal_mask,
)

from modeling.parallel_layers import (
    get_world_size,
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    TensorParallelEmbedding,
)


## Overwrite original functions
class RotaryEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, position_ids, cos_cache, sin_cache, past_key) -> torch.Tensor:
        seq_len = q.shape[-2]
        if past_key is not None:
            seq_len += past_key.shape[-2]
        # cos = cos_cache[:, :, :seq_len, ...].to(q.dtype)
        # sin = sin_cache[:, :, :seq_len, ...].to(q.dtype)
        cos = cos_cache.to(q.dtype)
        sin = sin_cache.to(q.dtype)

        cos = cos.reshape(cos.shape[2], cos.shape[3])
        sin = sin.reshape(sin.shape[2], sin.shape[3])
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        return q_embed

    @staticmethod
    def symbolic(g: torch.Graph, q, position_ids, cos_cache, sin_cache, past_key) -> (torch.Value, torch.Value):
        if past_key is None:
            return g.op("com.microsoft::RotaryEmbedding", q, position_ids, cos_cache, sin_cache)
        else:
            return g.op("com.microsoft::RotaryEmbedding", q, position_ids, cos_cache, sin_cache, past_key)


def new_mlp_init(self, config):
    super(LlamaMLP, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size

    ## Original
    # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.gate_proj = TensorParallelColumnLinear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = TensorParallelColumnLinear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = TensorParallelRowLinear(self.intermediate_size, self.hidden_size, bias=False)

    self.act_fn = ACT2FN[config.hidden_act]


def new_attention_init(self, config):
    super(LlamaAttention, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    # self.rope_theta = config.rope_theta

    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
            f" and `num_heads`: {self.num_heads})."
        )

    # Original
    # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
    # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
    # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
    # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    self.q_proj = TensorParallelColumnLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
    self.k_proj = TensorParallelColumnLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
    self.v_proj = TensorParallelColumnLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
    self.o_proj = TensorParallelRowLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    self._init_rope()


def attention_init_kv_cache(self, batch_size, device, dtype):
    """
    add kv cache buffer into attention_layer, this function needs to be called after parallel_split
    because it needs num_key_value_heads
    """
    self.key_cache = torch.zeros(
        (batch_size, self.num_key_value_heads, self.max_position_embeddings, self.head_dim), device=device, dtype=dtype
    )
    self.value_cache = torch.zeros(
        (batch_size, self.num_key_value_heads, self.max_position_embeddings, self.head_dim), device=device, dtype=dtype
    )


def new_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # use RotaryEmbedding for cudagraph, it need to handle cos_cache and sin_cache.
    query_states = RotaryEmbedding.apply(
        query_states,
        position_ids,
        self.rotary_emb.cos_cached,
        self.rotary_emb.sin_cached,
        past_key_value[0] if past_key_value is not None else None,
    )
    key_states = RotaryEmbedding.apply(
        key_states,
        position_ids,
        self.rotary_emb.cos_cached,
        self.rotary_emb.sin_cached,
        past_key_value[0] if past_key_value is not None else None,
    )
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if getattr(self, "key_cache", None) is not None:
        kv_position = position_ids[:, None, :, None].expand(
            key_states.shape[0], key_states.shape[1], position_ids.shape[1], key_states.shape[3]
        )
        self.key_cache.scatter_(2, kv_position, key_states)
        self.value_cache.scatter_(2, kv_position, value_states)

        key_states = self.key_cache
        value_states = self.value_cache
    else:
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # not need to check attn_weights and attention_mask's shape, because they are all use the max_seq_len
    # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    #    raise ValueError(
    #        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
    #        f" {attn_weights.size()}"
    #    )

    if attention_mask is not None:
        # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        #    raise ValueError(
        #        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        #    )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    ## Original
    # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


## New functions
def parallel_split(self):
    """
    parallel split for LlamaAttention
    """
    world_size = get_world_size()
    self.num_heads = self.num_heads // world_size
    nrep = 1
    if self.num_key_value_heads < world_size:
        assert world_size % self.num_key_value_heads == 0
        nrep = world_size // self.num_key_value_heads

    if nrep > 1:
        # repeat k, v proj to world_size * self.head_dim
        self.k_proj.repeat(nrep, self.num_key_value_heads, self.head_dim)
        self.v_proj.repeat(nrep, self.num_key_value_heads, self.head_dim)

    self.num_key_value_heads = (self.num_key_value_heads * nrep) // world_size
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads


def parallel_model(self):
    def _split_model(model):
        if isinstance(model, (TensorParallelColumnLinear, TensorParallelRowLinear, LlamaAttention)):
            model.parallel_split()
        for _, m in model._modules.items():
            _split_model(m)

    _split_model(self)


def addition_init(self, batch_size, device, dtype, max_seq_len):
    """
    This function is attached to LlamaForCausalLM, used to init kv_cache buffer and attention_mask buffer
    These buffers are used for cudagraph
    """

    def _init_fn(model):
        if isinstance(model, LlamaAttention):
            model.init_kv_cache(batch_size, device, dtype)
        for _, m in model._modules.items():
            _init_fn(m)

    _init_fn(self)

    self.decoder_attn_mask = torch.zeros((batch_size, 1, 1, max_seq_len), dtype=dtype, device=device)


def prepare_inputs(
    self, input_ids, position_ids=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    """
    This function is a replacement for prepare_inputs_for_generation
    It modifies attention_mask buffer for generation.
    """
    dtype = self.decoder_attn_mask.dtype
    min_value = torch.finfo(dtype).min
    if past_key_values is None:
        device = self.decoder_attn_mask.device
        max_seq_len = self.decoder_attn_mask.shape[3]
        bsz, seq_len = input_ids.shape

        self.decoder_attn_mask[:] = min_value
        self.decoder_attn_mask[:bsz, :, :, :seq_len] = 0.0

        mask = _make_causal_mask((bsz, seq_len), dtype, device)

        attention_mask = torch.full((bsz, 1, seq_len, max_seq_len), min_value, dtype=dtype, device=device)
        attention_mask[:, :, :, :seq_len] = mask

        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
        return model_inputs

    # for decoding phase, update mask at new position
    # position_ids shape is [b, seq_len]
    pos = position_ids[-1]
    self.decoder_attn_mask[:, :, :, pos] = 0.0

    model_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": self.decoder_attn_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
    }
    return model_inputs


def prepare_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    """
    this is a hook for LlamaModel._prepare_decoder_attention_mask
    attention_mask has been modified by prepare_inputs, so here directly return it
    """
    if len(attention_mask.shape) == 2:
        # fallback to origin prepare attention_mask
        return self._prepare_decoder_attention_mask_backup(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

    return attention_mask


LlamaModel._prepare_decoder_attention_mask_backup = LlamaModel._prepare_decoder_attention_mask
LlamaModel._prepare_decoder_attention_mask = prepare_attention_mask
LlamaAttention.init_kv_cache = attention_init_kv_cache
LlamaAttention.forward = new_attention_forward
LlamaForCausalLM.addition_init = addition_init
LlamaForCausalLM.prepare_inputs = prepare_inputs

if get_world_size() > 1:
    LlamaMLP.__init__ = new_mlp_init
    LlamaAttention.__init__ = new_attention_init
    LlamaAttention.parallel_split = parallel_split
    LlamaForCausalLM.parallel_model = parallel_model

    print("[modeling.patching_llama] Patching complete!")
else:
    print("[modeling.patching_llama] Only LlamaAttention.forward patched.")
