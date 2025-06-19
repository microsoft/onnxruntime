# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import onnx
import torch
from parameterized import parameterized
from parity_utilities import find_transformers_source
from transformers import EncoderDecoderCache

if find_transformers_source():
    from fusion_options import FusionOptions
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_model


# Dummy constants smaller than openai/whisper-tiny
class WhisperConfig:
    def __init__(self):
        # Hugging Face attribute names
        self.hidden_size = 10
        self.num_heads = 2
        self.head_dim = self.hidden_size // self.num_heads
        self.d_model = self.embed_dim = self.hidden_size
        self.encoder_sequence_length = 20
        self.encoder_ffn_dim = 10
        self.decoder_ffn_dim = 10

        # OpenAI attribute names
        self.n_state = self.hidden_size
        self.n_head = self.num_heads
        self.n_mlp = self.encoder_ffn_dim


# From https://github.com/huggingface/transformers/blob/31f8a0fe8a7e2db1ee30bf32ed5976cd11f3283c/src/transformers/models/whisper/modeling_whisper.py#L222
class WhisperHFAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = WhisperConfig()

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.layer_idx = 0

        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        past_key_value: tuple[tuple[torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""
        is_updated = past_key_value is not None
        past_key_value = EncoderDecoderCache.from_legacy_cache(past_key_value)
        past_key_value.is_updated[self.layer_idx] = is_updated

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim)
        query_states = query_states.transpose(1, 2).contiguous()

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim)
            value_states = self.v_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim)
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        past_key_value = past_key_value.to_legacy_cache()
        return attn_output, past_key_value


# From https://github.com/huggingface/transformers/blob/31f8a0fe8a7e2db1ee30bf32ed5976cd11f3283c/src/transformers/models/whisper/modeling_whisper.py#L583
class WhisperHFEncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = WhisperConfig()
        self.embed_dim = config.d_model

        self.self_attn = WhisperHFAttention()
        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.activation_fn = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = torch.nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
        """
        hidden_states += 1  # Add fake add to help with fusion testing

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        return outputs


# From https://github.com/huggingface/transformers/blob/31f8a0fe8a7e2db1ee30bf32ed5976cd11f3283c/src/transformers/models/whisper/modeling_whisper.py#L651
class WhisperHFDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = WhisperConfig()
        self.embed_dim = config.d_model

        self.self_attn = WhisperHFAttention()
        self.activation_fn = torch.nn.GELU()

        self.self_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperHFAttention()
        self.encoder_attn_layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.fc1 = torch.nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = torch.nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        cross_attn_layer_head_mask: torch.Tensor | None = None,
        past_key_value: tuple[tuple[torch.Tensor]] | None = None,
        use_cache: bool | None = True,
        cache_position: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
        """
        hidden_states += 1  # Add fake add to help with fusion testing
        batch_size, target_length = attention_mask.shape  # Get shape to create 4D attention mask
        sequence_length = hidden_states.size(1)  # Get shape to create 4D attention mask

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask[:, None, None, :].expand(batch_size, 1, sequence_length, target_length),
            layer_head_mask=layer_head_mask,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
            )
            hidden_states = residual + hidden_states

            # add cross-attn to positions 1 of present_key_value tuple
            if past_key_value is None:
                # Skip if cross-attention has past KV cache inputs since the outputs are identical
                present_key_value = (present_key_value, cross_attn_present_key_value)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# From https://github.com/openai/whisper/blob/dd985ac4b90cafeef8712f2998d62c59c3e62d22/whisper/model.py#L44
class WhisperOAILinear(torch.nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


# From https://github.com/openai/whisper/blob/423492dda7806206abe56bdfe427c1096473a020/whisper/model.py#L62
class WhisperOAIAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = WhisperConfig()
        self.n_head = config.n_head
        self.query = WhisperOAILinear(config.n_state, config.n_state)
        self.key = WhisperOAILinear(config.n_state, config.n_state, bias=False)
        self.value = WhisperOAILinear(config.n_state, config.n_state)
        self.out = WhisperOAILinear(config.n_state, config.n_state)

    def forward(
        self,
        x: torch.Tensor,
        xa: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor] | None = None,
    ):
        q = self.query(x)
        present_k, present_v = None, None

        if kv_cache is None or xa is None:
            # If xa == None: self-attention without KV cache inputs
            # If xa != None: cross-attention without KV cache inputs
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)

            if mask is not None and kv_cache is not None:
                # Self-attention with KV cache inputs and outputs
                past_k = kv_cache[0]
                past_k = past_k.transpose(1, 2)
                past_k = past_k.reshape(past_k.shape[:2] + (-1,))
                past_v = kv_cache[1]
                past_v = past_v.transpose(1, 2)
                past_v = past_v.reshape(past_v.shape[:2] + (-1,))

                present_k = torch.cat([past_k, k], dim=1)
                present_v = torch.cat([past_v, v], dim=1)

                present_k = present_k.reshape(present_k.shape[:2] + (-1, self.n_head)).transpose(1, 2)
                present_v = present_v.reshape(v.shape[:2] + (-1, self.n_head)).transpose(1, 2)
        else:
            # Cross-attention with KV cache inputs
            past_k = kv_cache[0]
            past_k = past_k.transpose(1, 2)
            past_k = past_k.reshape(past_k.shape[:2] + (-1,))
            past_v = kv_cache[1]
            past_v = past_v.transpose(1, 2)
            past_v = past_v.reshape(past_v.shape[:2] + (-1,))
            k = past_k
            v = past_v

        n_batch, n_ctx, n_state = q.shape
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        wv, qk = self.qkv_attention(q, k, v, mask, n_ctx, n_state)
        o = self.out(wv)

        if mask is None and kv_cache is not None:
            # Cross-attention with KV cache inputs
            return o, None, None

        if mask is not None and kv_cache is not None:
            # Self-attention with KV cache inputs and outputs
            return o, present_k, present_v

        return o, k, v

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        n_ctx: int,
        n_state: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        scale = (n_state // self.n_head) ** -0.25

        qk = (q * scale) @ (k * scale)
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = torch.nn.functional.softmax(qk, dim=-1)
        out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        qk = qk.detach()

        return out, qk


# From https://github.com/openai/whisper/blob/dd985ac4b90cafeef8712f2998d62c59c3e62d22/whisper/model.py#L142
class WhisperOAIResidualAttentionBlock(torch.nn.Module):
    def __init__(self, cross_attention: bool = False):
        super().__init__()
        config = WhisperConfig()

        self.attn = WhisperOAIAttention()
        self.attn_ln = torch.nn.LayerNorm(config.n_state)

        self.cross_attn = WhisperOAIAttention() if cross_attention else None
        self.cross_attn_ln = torch.nn.LayerNorm(config.n_state) if cross_attention else None

        self.mlp = torch.nn.Sequential(
            WhisperOAILinear(config.n_state, config.n_mlp),
            torch.nn.GELU(),
            WhisperOAILinear(config.n_mlp, config.n_state),
        )
        self.mlp_ln = torch.nn.LayerNorm(config.n_state)

    def forward(
        self,
        x: torch.Tensor,
        xa: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor] | None = None,
    ):
        x += 1  # Add fake add to help with fusion testing

        self_attn_output, self_k, self_v = self.attn(
            self.attn_ln(x), mask=mask, kv_cache=(kv_cache[:2] if kv_cache is not None else kv_cache)
        )
        x = x + self_attn_output
        if self.cross_attn:
            cross_attn_output, cross_k, cross_v = self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=(kv_cache[2:] if kv_cache is not None else kv_cache)
            )
            x = x + cross_attn_output
        else:
            self_k = self_v = cross_k = cross_v = None  # Set to none when creating encoder model's attention block
        x = x + self.mlp(self.mlp_ln(x))
        return x, (self_k, self_v, cross_k, cross_v)


class TestFusion(unittest.TestCase):
    def verify_fusion(self, optimized_model, expected_model_filename):
        optimized_model.topological_sort(is_deterministic=True)

        expected_model_path = os.path.join(
            os.path.dirname(__file__), "test_data", "models", "whisper", expected_model_filename
        )
        expected_model = OnnxModel(onnx.load(expected_model_path))
        expected_model.topological_sort(is_deterministic=True)

        nodes = optimized_model.model.graph.node
        self.assertEqual(len(nodes), len(expected_model.model.graph.node))

        for i in range(len(nodes)):
            self.assertEqual(nodes[i], expected_model.model.graph.node[i])

        for expected_initializer in expected_model.model.graph.initializer:
            self.assertTrue(
                OnnxModel.has_same_value(
                    optimized_model.get_initializer(expected_initializer.name), expected_initializer
                )
            )

    def export(self, model, inputs, input_names, output_names, dynamic_axes):
        torch.onnx.export(
            model,
            args=inputs,
            f=os.path.join(os.path.dirname(__file__), "export.onnx"),
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
        )

    def setUp(self):
        # Reset the seed to 0 so that the tensor weights stay the same for each test case
        # whether FP16 or FP32 is tested in a CI
        torch.manual_seed(0)

        self.config = WhisperConfig()
        self.optimization_options = FusionOptions("bart")
        self.optimization_options.use_multi_head_attention = True

        self.batch_size = 2
        self.sequence_length = 10

    def postSetUp(self, precision, split_bias=False):  # noqa: N802
        use_fp16 = precision == "fp16"
        self.device = torch.device("cuda" if use_fp16 else "cpu")
        self.torch_dtype = torch.float16 if use_fp16 else torch.float32
        self.optimization_options.disable_multi_head_attention_bias = split_bias

    def tearDown(self):
        path = os.path.join(os.path.dirname(__file__), "export.onnx")
        if os.path.exists(path):
            os.remove(path)

    @parameterized.expand(
        [
            ("fp16", "cuda"),
            ("fp32", "cpu"),
        ]
    )
    def test_hf_whisper_encoder_self_attention(self, precision, ep):
        if ep == "cuda" and not torch.cuda.is_available():
            return
        self.postSetUp(precision)
        model = WhisperHFEncoderLayer().to(dtype=self.torch_dtype, device=self.device)

        hidden_states = torch.randn(
            self.batch_size, self.sequence_length, self.config.embed_dim, device=self.device, dtype=self.torch_dtype
        )
        inputs = (hidden_states,)
        self.export(
            model, inputs, input_names=["input_hidden_states"], output_names=["output_hidden_states"], dynamic_axes={}
        )

        original_model = onnx.load(os.path.join(os.path.dirname(__file__), "export.onnx"))
        optimized_model = optimize_model(
            original_model,
            model_type="bart",
            num_heads=self.config.num_heads,
            hidden_size=self.config.embed_dim,
            optimization_options=self.optimization_options,
            opt_level=0,
            use_gpu=True,
            only_onnxruntime=False,
        )
        name = f"hf_{precision}_encoder_self_attention.onnx"
        # optimized_model.save_model_to_file(name)  # Uncomment for debugging purposes
        self.verify_fusion(optimized_model, name)

    @parameterized.expand(
        [
            ("fp16", "cuda", False),
            ("fp16", "cuda", True),
            ("fp32", "cpu", False),
            ("fp32", "cpu", True),
        ]
    )
    def test_hf_whisper_decoder_no_past(self, precision, ep, split_bias):
        if ep == "cuda" and not torch.cuda.is_available():
            return
        self.postSetUp(precision, split_bias)
        model = WhisperHFDecoderLayer().to(dtype=self.torch_dtype, device=self.device)

        hidden_states = torch.randn(
            self.batch_size, self.sequence_length, self.config.embed_dim, device=self.device, dtype=self.torch_dtype
        )
        attention_mask = torch.ones(self.batch_size, self.sequence_length, device=self.device, dtype=self.torch_dtype)
        encoder_hidden_states = torch.randn(
            self.batch_size,
            self.config.encoder_sequence_length,
            self.config.embed_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        inputs = (
            hidden_states,
            attention_mask,
            encoder_hidden_states,
        )
        self.export(
            model,
            inputs,
            input_names=["input_hidden_states", "attention_mask", "encoder_hidden_states"],
            output_names=[
                "output_hidden_states",
                "present_key_self",
                "present_value_self",
                "present_key_cross",
                "present_value_cross",
            ],
            dynamic_axes={},
        )

        original_model = onnx.load(os.path.join(os.path.dirname(__file__), "export.onnx"))
        optimized_model = optimize_model(
            original_model,
            model_type="bart",
            num_heads=self.config.num_heads,
            hidden_size=self.config.embed_dim,
            optimization_options=self.optimization_options,
            opt_level=0,
            use_gpu=True,
            only_onnxruntime=False,
        )
        name = f"hf_{precision}_decoder_attention_no_past{'_split_bias' if split_bias else ''}.onnx"
        # optimized_model.save_model_to_file(name)  # Uncomment for debugging purposes
        self.verify_fusion(optimized_model, name)

    @parameterized.expand(
        [
            ("fp16", "cuda", False),
            ("fp16", "cuda", True),
            ("fp32", "cpu", False),
            ("fp32", "cpu", True),
        ]
    )
    def test_hf_whisper_decoder_with_past(self, precision, ep, split_bias):
        if ep == "cuda" and not torch.cuda.is_available():
            return
        self.postSetUp(precision, split_bias)
        model = WhisperHFDecoderLayer().to(dtype=self.torch_dtype, device=self.device)

        hidden_states = torch.randn(
            self.batch_size, 1, self.config.embed_dim, device=self.device, dtype=self.torch_dtype
        )
        attention_mask = torch.ones(
            self.batch_size, self.sequence_length + 1, device=self.device, dtype=self.torch_dtype
        )
        encoder_hidden_states = torch.randn(
            self.batch_size,
            self.config.encoder_sequence_length,
            self.config.embed_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        past_key_self = torch.randn(
            self.batch_size,
            self.config.num_heads,
            self.sequence_length,
            self.config.head_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        past_value_self = torch.randn(
            self.batch_size,
            self.config.num_heads,
            self.sequence_length,
            self.config.head_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        past_key_cross = torch.randn(
            self.batch_size,
            self.config.num_heads,
            self.config.encoder_sequence_length,
            self.config.head_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        past_value_cross = torch.randn(
            self.batch_size,
            self.config.num_heads,
            self.config.encoder_sequence_length,
            self.config.head_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )

        # past_key_values is of shape (num_layers) where each element is of shape (4)
        #
        # Ex:
        # past_key_values = (layer_0_tuple, layer_1_tuple,)
        # layer_0_tuple = (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0,)
        # layer_1_tuple = (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1,)
        past_key_values = (
            (
                past_key_self,
                past_value_self,
                past_key_cross,
                past_value_cross,
            ),
        )

        inputs = (
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            None,
            None,
            None,
            past_key_values,
        )
        self.export(
            model,
            inputs,
            input_names=[
                "input_hidden_states",
                "attention_mask",
                "encoder_hidden_states",
                "past_key_self",
                "past_value_self",
                "past_key_cross",
                "past_value_cross",
            ],
            output_names=["output_hidden_states", "present_key_self", "present_value_self"],
            dynamic_axes={},
        )

        original_model = onnx.load(os.path.join(os.path.dirname(__file__), "export.onnx"))
        optimized_model = optimize_model(
            original_model,
            model_type="bart",
            num_heads=self.config.num_heads,
            hidden_size=self.config.embed_dim,
            optimization_options=self.optimization_options,
            opt_level=0,
            use_gpu=True,
            only_onnxruntime=False,
        )
        name = f"hf_{precision}_decoder_attention_with_past{'_split_bias' if split_bias else ''}.onnx"
        # optimized_model.save_model_to_file(name)  # Uncomment for debugging purposes
        self.verify_fusion(optimized_model, name)

    @parameterized.expand(
        [
            ("fp16", "cuda"),
            ("fp32", "cpu"),
        ]
    )
    def test_oai_whisper_encoder_self_attention(self, precision, ep):
        if ep == "cuda" and not torch.cuda.is_available():
            return
        self.postSetUp(precision)
        model = WhisperOAIResidualAttentionBlock().to(dtype=self.torch_dtype, device=self.device)

        hidden_states = torch.randn(
            self.batch_size, self.sequence_length, self.config.embed_dim, device=self.device, dtype=self.torch_dtype
        )
        inputs = (hidden_states,)
        self.export(
            model, inputs, input_names=["input_hidden_states"], output_names=["output_hidden_states"], dynamic_axes={}
        )

        original_model = onnx.load(os.path.join(os.path.dirname(__file__), "export.onnx"))
        optimized_model = optimize_model(
            original_model,
            model_type="bart",
            num_heads=self.config.num_heads,
            hidden_size=self.config.embed_dim,
            optimization_options=self.optimization_options,
            opt_level=0,
            use_gpu=True,
            only_onnxruntime=False,
        )
        name = f"oai_{precision}_encoder_self_attention.onnx"
        # optimized_model.save_model_to_file(name)  # Uncomment for debugging purposes
        self.verify_fusion(optimized_model, name)

    @parameterized.expand(
        [
            ("fp16", "cuda", False),
            ("fp16", "cuda", True),
            ("fp32", "cpu", False),
            ("fp32", "cpu", True),
        ]
    )
    def test_oai_whisper_decoder_no_past(self, precision, ep, split_bias):
        if ep == "cuda" and not torch.cuda.is_available():
            return
        self.postSetUp(precision, split_bias)
        model = WhisperOAIResidualAttentionBlock(cross_attention=True).to(dtype=self.torch_dtype, device=self.device)

        hidden_states = torch.randn(
            self.batch_size, self.sequence_length, self.config.embed_dim, device=self.device, dtype=self.torch_dtype
        )
        encoder_hidden_states = torch.randn(
            self.batch_size,
            self.config.encoder_sequence_length,
            self.config.embed_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        attention_mask = torch.ones(
            self.sequence_length, self.sequence_length, device=self.device, dtype=self.torch_dtype
        )
        inputs = (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
        )
        self.export(
            model,
            inputs,
            input_names=[
                "input_hidden_states",
                "encoder_hidden_states",
                "attention_mask",
            ],
            output_names=[
                "output_hidden_states",
                "present_key_self",
                "present_value_self",
                "present_key_cross",
                "present_value_cross",
            ],
            dynamic_axes={},
        )

        original_model = onnx.load(os.path.join(os.path.dirname(__file__), "export.onnx"))
        optimized_model = optimize_model(
            original_model,
            model_type="bart",
            num_heads=self.config.num_heads,
            hidden_size=self.config.embed_dim,
            optimization_options=self.optimization_options,
            opt_level=0,
            use_gpu=True,
            only_onnxruntime=False,
        )
        name = f"oai_{precision}_decoder_attention_no_past{'_split_bias' if split_bias else ''}.onnx"
        # optimized_model.save_model_to_file(name)  # Uncomment for debugging purposes
        self.verify_fusion(optimized_model, name)

    @parameterized.expand(
        [
            ("fp16", "cuda", False),
            ("fp16", "cuda", True),
            ("fp32", "cpu", False),
            ("fp32", "cpu", True),
        ]
    )
    def test_oai_whisper_decoder_with_past(self, precision, ep, split_bias):
        if ep == "cuda" and not torch.cuda.is_available():
            return
        self.postSetUp(precision, split_bias)
        model = WhisperOAIResidualAttentionBlock(cross_attention=True).to(dtype=self.torch_dtype, device=self.device)

        hidden_states = torch.randn(
            self.batch_size, 1, self.config.embed_dim, device=self.device, dtype=self.torch_dtype
        )
        encoder_hidden_states = torch.randn(
            self.batch_size,
            self.config.encoder_sequence_length,
            self.config.embed_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        attention_mask = torch.ones(1, 1, device=self.device, dtype=self.torch_dtype)
        past_key_self = torch.randn(
            self.batch_size,
            self.config.num_heads,
            self.sequence_length,
            self.config.head_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        past_value_self = torch.randn(
            self.batch_size,
            self.config.num_heads,
            self.sequence_length,
            self.config.head_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        past_key_cross = torch.randn(
            self.batch_size,
            self.config.num_heads,
            self.config.encoder_sequence_length,
            self.config.head_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        past_value_cross = torch.randn(
            self.batch_size,
            self.config.num_heads,
            self.config.encoder_sequence_length,
            self.config.head_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )

        # past_key_values is of shape (num_layers) where each element is a past key/value
        #
        # Ex:
        # past_key_values = (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0,)
        past_key_values = (
            past_key_self,
            past_value_self,
            past_key_cross,
            past_value_cross,
        )

        inputs = (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            past_key_values,
        )
        self.export(
            model,
            inputs,
            input_names=[
                "input_hidden_states",
                "encoder_hidden_states",
                "attention_mask",
                "past_key_self",
                "past_value_self",
                "past_key_cross",
                "past_value_cross",
            ],
            output_names=["output_hidden_states", "present_key_self", "present_value_self"],
            dynamic_axes={},
        )

        original_model = onnx.load(os.path.join(os.path.dirname(__file__), "export.onnx"))
        optimized_model = optimize_model(
            original_model,
            model_type="bart",
            num_heads=self.config.num_heads,
            hidden_size=self.config.embed_dim,
            optimization_options=self.optimization_options,
            opt_level=0,
            use_gpu=True,
            only_onnxruntime=False,
        )
        name = f"oai_{precision}_decoder_attention_with_past{'_split_bias' if split_bias else ''}.onnx"
        # optimized_model.save_model_to_file(name)  # Uncomment for debugging purposes
        self.verify_fusion(optimized_model, name)


if __name__ == "__main__":
    unittest.main()
