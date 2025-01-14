# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import onnx
import torch
from parity_utilities import find_transformers_source

if find_transformers_source():
    from fusion_options import FusionOptions
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_model


# From https://github.com/huggingface/transformers/blob/34f76bb62b915b43617aa88557aea97840e163f0/src/transformers/activations.py#L90
class PhiVCLIPQuickGelu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


# Line-by-line calculation of https://github.com/huggingface/transformers/blob/34f76bb62b915b43617aa88557aea97840e163f0/src/transformers/models/clip/modeling_clip.py#L613
class PhiVCLIPLayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(20)).to(torch.float16).detach()
        self.bias = torch.nn.Parameter(torch.ones(20)).to(torch.float16).detach()
        self.eps = 1e-05

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        diff = (x - mean).to(torch.float64)
        variance = diff.pow(2).mean(-1, keepdim=True)
        x = diff / torch.sqrt(variance + self.eps)
        x = x.to(torch.float16) * self.weight + self.bias
        return x


# From https://github.com/huggingface/transformers/blob/34f76bb62b915b43617aa88557aea97840e163f0/src/transformers/models/clip/modeling_clip.py#L300
class PhiVCLIPAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 20
        self.num_heads = 2
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5

        self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)

        self.k_proj.weight.data.fill_(1)
        self.k_proj.bias.data.fill_(1)
        self.v_proj.weight.data.fill_(1)
        self.v_proj.bias.data.fill_(1)
        self.q_proj.weight.data.fill_(1)
        self.q_proj.bias.data.fill_(1)
        self.out_proj.weight.data.fill_(1)
        self.out_proj.bias.data.fill_(1)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=False,
    ):
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = torch.nn.functional.dropout(attn_weights, p=0, training=False)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class PhiVCLIPAttentionAndLayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = PhiVCLIPAttention()
        self.ln = torch.nn.LayerNorm(20, eps=1e-05)

    def forward(self, x):
        #      SkipLayerNorm ------+
        #            |             |
        #        Attention         |
        #            |             |
        #          MatMul          |
        #            |             |
        #      SkipLayerNorm ------+

        # SkipLayerNorm
        x = x + x
        x = self.ln(x)
        residual = x

        # Attention + MatMul
        x = self.attn(x)

        # SkipLayerNorm
        x = residual + x
        x = self.ln(x)
        return x


class TestFusion(unittest.TestCase):
    def verify_fusion(self, optimized_model, expected_model_filename):
        optimized_model.topological_sort(is_deterministic=True)

        expected_model_path = os.path.join(os.path.dirname(__file__), "test_data", "models", expected_model_filename)
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

    def export(self, model, inputs):
        torch.onnx.export(
            model,
            args=inputs,
            f=os.path.join(os.path.dirname(__file__), "export.onnx"),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
        )

    def tearDown(self):
        path = os.path.join(os.path.dirname(__file__), "export.onnx")
        if os.path.exists(path):
            os.remove(path)

    def test_phi_vision_layernorm(self):
        if not torch.cuda.is_available():
            return
        model = PhiVCLIPLayerNorm()
        inputs = (torch.randn(1, 2, 20).to(torch.float16),)
        self.export(model, inputs)
        original_model = onnx.load(os.path.join(os.path.dirname(__file__), "export.onnx"))
        options = FusionOptions("clip")
        optimized_model = optimize_model(
            original_model,
            model_type="clip",
            num_heads=2,
            hidden_size=20,
            optimization_options=options,
            opt_level=0,
            use_gpu=True,
        )
        self.verify_fusion(optimized_model, "phi-3.5-v-instruct-vision-layernorm.onnx")

    def test_phi_vision_quickgelu(self):
        model = PhiVCLIPQuickGelu()
        inputs = (torch.randn(1, 2, 20),)
        self.export(model, inputs)
        original_model = onnx.load(os.path.join(os.path.dirname(__file__), "export.onnx"))
        options = FusionOptions("clip")
        optimized_model = optimize_model(
            original_model, model_type="clip", num_heads=2, hidden_size=20, optimization_options=options, opt_level=0
        )
        self.verify_fusion(optimized_model, "phi-3.5-v-instruct-vision-quickgelu.onnx")

    def test_phi_vision_attention(self):
        model = PhiVCLIPAttentionAndLayerNorm()
        inputs = (torch.randn(1, 2, 20),)
        self.export(model, inputs)
        original_model = onnx.load(os.path.join(os.path.dirname(__file__), "export.onnx"))
        options = FusionOptions("clip")
        optimized_model = optimize_model(
            original_model, model_type="clip", num_heads=2, hidden_size=20, optimization_options=options, opt_level=0
        )
        self.verify_fusion(optimized_model, "phi-3.5-v-instruct-vision-attention.onnx")


if __name__ == "__main__":
    unittest.main()
