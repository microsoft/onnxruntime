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

if find_transformers_source():
    from dynamo_onnx_helper import DynamoOnnxHelper
    from fusion_options import FusionOptions
    from onnx_model import OnnxModel
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.dynamo_onnx_helper import DynamoOnnxHelper
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.optimizer import optimize_model


# https://github.com/huggingface/transformers/blob/af9b2eaa54c150741f298d6db939af6328e1dc38/src/transformers/models/siglip/modeling_siglip.py#L363
class SiglipAttention(torch.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self):
        super().__init__()
        self.embed_dim = 20
        self.num_heads = 2
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5

        self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)

        self.k_proj.weight.data.fill_(1)
        self.v_proj.weight.data.fill_(1)
        self.q_proj.weight.data.fill_(1)
        self.out_proj.weight.data.fill_(1)
        self.k_proj.bias.data.fill_(1)
        self.v_proj.bias.data.fill_(1)
        self.q_proj.bias.data.fill_(1)
        self.out_proj.bias.data.fill_(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Gemma3VSIGLIPAttentionAndLayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SiglipAttention()
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
        x, _ = self.attn(x)

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
                    optimized_model.get_initializer(expected_initializer.name),
                    expected_initializer,
                )
            )

    def export(self, model, inputs) -> onnx.ModelProto:
        with torch.no_grad():
            onnx_program = torch.onnx.export(
                model,
                args=inputs,
                # f=os.path.join(os.path.dirname(__file__), "export.onnx"),
                dynamo=True,
                optimize=True,
            )
        return onnx_program.model_proto  # type: ignore

    def tearDown(self):
        paths = [
            os.path.join(os.path.dirname(__file__), "export.onnx"),
            os.path.join(os.path.dirname(__file__), "export.onnx.data"),
        ]
        for path in paths:
            if os.path.exists(path):
                os.remove(path)

    @unittest.skip(reason="Fails with ONNX 1.18.0")
    @parameterized.expand(
        [
            (torch.float32, "gemma3-vision-attention_fp32.onnx"),
            (torch.float16, "gemma3-vision-attention_fp16.onnx"),
        ]
    )
    def test_gemma3_vision_attention(self, dtype, model_name):
        model = Gemma3VSIGLIPAttentionAndLayerNorm().eval().to(dtype)
        inputs = (torch.randn(1, 2, 20, dtype=dtype),)
        original_model = self.export(model, inputs)

        # TODO(titaiwang): Upstream these processings to onnxscript pass
        onnx_model_wrapper = DynamoOnnxHelper(original_model)
        onnx_model_wrapper.convert_constants_to_initializers()
        onnx_model_wrapper.clear_metadata()
        model_path = os.path.join(os.path.dirname(__file__), "export.onnx")
        onnx_model_wrapper.model.save_model_to_file(
            model_path,
            use_external_data_format=True,
            all_tensors_to_one_file=True,
            convert_attribute=True,
        )

        options = FusionOptions("clip")
        optimized_model = optimize_model(
            model_path,
            model_type="clip",
            num_heads=2,
            hidden_size=20,
            optimization_options=options,
            opt_level=0,
        )
        self.verify_fusion(optimized_model, model_name)


if __name__ == "__main__":
    unittest.main()
