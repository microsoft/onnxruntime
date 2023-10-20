# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


# Notes
# 1) The test cases in this file are for the following LLaMA-2 scenarios:
# - Microsoft rotary embeddings with interleaved = True
#   - Prompt generation
#   - Token generation
# - Hugging Face rotary embeddings (equal to Microsoft rotary embeddings with interleaved = False)
#   - Prompt generation
#   - Token generation
#
# 2) Shapes of position ids in ORT and `interleaved` for LLaMA-2 scenarios:
# - Microsoft model: When shape of position ids == (1), interleaved = True
# - Hugging Face model: When shape of position ids == (batch_size, sequence_length), interleaved = False


import unittest
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from onnx import TensorProto, helper

import onnxruntime as ort


class SampleInputConfig:
    def __init__(
        self,
        batch_size=2,
        sequence_length=8,
        num_heads=4,
        head_size=6,
        max_sequence_length=16,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.head_size = head_size
        self.hidden_size = self.num_heads * self.head_size
        self.max_sequence_length = max_sequence_length


# LLaMA Hugging Face model
class LlamaHFRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device="cpu"):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def get_cos_sin_cache(self, seq_len=None, device=torch.device("cpu"), dtype=torch.float32):  # noqa: B008
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=dtype),
        )

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope_bnsh(self, x, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        x_embed = (x * cos) + (self.rotate_half(x) * sin)
        return x_embed

    def apply_rope_bsnh(self, x, cos, sin, position_ids):
        # Two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze()  # [seq_len, dim]
        sin = sin.squeeze()  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        x_embed = (x * cos) + (self.rotate_half(x) * sin)
        return x_embed

    def forward(self, x, cos, sin, pos_ids, x_format="bnsh"):
        if x_format == "bnsh":
            return self.apply_rope_bnsh(x, cos, sin, pos_ids)
        return self.apply_rope_bsnh(x, cos, sin, pos_ids)


# LLaMA Microsoft model
class LlamaMSRotaryEmbedding(nn.Module):
    def __init__(self, hidden_size, num_heads, max_sequence_length):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length

    def get_cos_sin_cache(self, theta=10000.0, head_scale=1.0, device="cpu", dtype=torch.float32):
        hidden_size = self.hidden_size
        n_heads = self.num_heads
        max_seq_len = self.max_sequence_length

        # Precalculate rotary matrices for the sequence
        # According to "Attention Is All You Need", theta_i = 10000 ^ (2 * (i - 1)/dim), i in [1, 2, ..., dim//2]
        head_dim = head_scale * hidden_size / n_heads

        pos = torch.arange(0, 2 * (head_dim // 2), step=2, device=device, dtype=dtype)
        freqs = 1.0 / (theta ** (pos / head_dim))

        idx = torch.arange(max_seq_len, device=freqs.device)
        freqs = torch.outer(idx, freqs)

        cos = torch.reshape(torch.cos(freqs), [1, max_seq_len, 1, -1])
        sin = torch.reshape(torch.sin(freqs), [1, max_seq_len, 1, -1])
        dtype = torch.get_default_dtype()

        return cos.to(dtype), sin.to(dtype)

    def rotate_tensor(
        self,
        x: torch.Tensor,  # BxSxNxH
        cos: torch.Tensor,  # 1xSx1x(H/2)
        sin: torch.Tensor,  # 1xSx1x(H/2)
        pos: int,
        interleaved: bool,
    ):
        # Dimension of x is [batch_size, seq_len, n_heads, head_dim]
        rot_dim = 2 * cos.shape[3]

        # Dolly requires partial rotation
        x_rot = x[:, :, :, :rot_dim]

        if interleaved:
            x1 = x_rot[:, :, :, 0::2]
            x2 = x_rot[:, :, :, 1::2]
        else:
            half = x_rot.shape[-1] // 2
            x1 = x[:, :, :, 0:half]
            x2 = x[:, :, :, half : 2 * half]

        seq_len = x.shape[1]
        cos_x = cos[:, pos : pos + seq_len, :, :]
        sin_x = sin[:, pos : pos + seq_len, :, :]

        # cos_x: (1, S, 1, H/2)
        # sin_x: (1, S, 1, H/2)
        # x1: (B, S, N, H/2)
        # x2: (B, S, N, H/2)
        real = cos_x * x1 - sin_x * x2
        imag = sin_x * x1 + cos_x * x2

        if interleaved:
            x_rot[:, :, :, 0::2] = real
            x_rot[:, :, :, 1::2] = imag
        else:
            x_rot = torch.cat((real, imag), dim=-1)

        return torch.cat((x_rot, x[:, :, :, rot_dim:]), dim=-1)

    def forward(self, x, cos, sin, pos, interleaved):
        return self.rotate_tensor(x, cos, sin, pos, interleaved)


class TestLlamaRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        self.config = SampleInputConfig()
        self.llama_hf = LlamaHFRotaryEmbedding(self.config.head_size, self.config.max_sequence_length)
        self.llama_ms = LlamaMSRotaryEmbedding(
            self.config.hidden_size, self.config.num_heads, self.config.max_sequence_length
        )

        seed = 2
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_printoptions(sci_mode=False)

    def create_onnx_graph(self, x_shape, pos_shape, cos, sin, interleaved):
        inputs = [
            helper.make_tensor_value_info(
                name="input",
                elem_type=TensorProto.FLOAT,
                shape=list(x_shape),
            ),
            helper.make_tensor_value_info(
                name="position_ids",
                elem_type=TensorProto.INT64,
                shape=list(pos_shape),
            ),
        ]
        outputs = [
            helper.make_tensor_value_info(
                name="output",
                elem_type=TensorProto.FLOAT,
                shape=list(x_shape),
            ),
        ]

        initializers = [
            helper.make_tensor(
                name="cos_cache",
                data_type=TensorProto.FLOAT,
                dims=list(torch.squeeze(cos).shape),
                vals=cos.flatten().tolist(),
            ),
            helper.make_tensor(
                name="sin_cache",
                data_type=TensorProto.FLOAT,
                dims=list(torch.squeeze(sin).shape),
                vals=sin.flatten().tolist(),
            ),
        ]
        nodes = [
            helper.make_node(
                op_type="RotaryEmbedding",
                inputs=["input", "position_ids", "cos_cache", "sin_cache"],
                outputs=["output"],
                interleaved=interleaved,
                name="RotaryEmbedding_0",
                domain="com.microsoft",
            ),
        ]

        graph = helper.make_graph(
            nodes=nodes,
            name="RotaryEmbedding_Graph",
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )
        opset_import = helper.make_opsetid(domain="com.microsoft", version=1)
        model = helper.make_model(graph, opset_imports=[opset_import])
        return model.SerializeToString()

    def get_eps(self):
        eps = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        return list(filter(lambda ep: ep in ort.get_available_providers(), eps))

    def run_ort_ep_tests(self, onnx_graph, inputs_ort, expected_output_bsnh):
        eps = self.get_eps()
        for ep in eps:
            sess = ort.InferenceSession(onnx_graph, providers=[ep])
            output_ort = sess.run(None, inputs_ort)[0]
            output_ort = output_ort.reshape(
                (self.config.batch_size, inputs_ort["input"].shape[1], self.config.num_heads, self.config.head_size)
            )

            # Compare outputs as BxSxNxH
            self.assertTrue(np.allclose(expected_output_bsnh, output_ort))

    # apply_rope(x_bnsh) == apply_rope(x_bsnh).transpose(1,2)
    def test_hf_bnsh_and_hf_bsnh(self):
        x_bnsh = torch.randn(
            self.config.batch_size, self.config.num_heads, self.config.sequence_length, self.config.head_size
        )
        cos_hf, sin_hf = self.llama_hf.get_cos_sin_cache(self.config.sequence_length)
        pos_hf = torch.stack([torch.arange(0, self.config.sequence_length) for _ in range(self.config.batch_size)])

        x_bnsh_after_rope = self.llama_hf(x_bnsh, cos_hf, sin_hf, pos_hf)  # output is BxNxSxH
        x_bsnh_after_rope = self.llama_hf(
            x_bnsh.transpose(1, 2), cos_hf.transpose(1, 2), sin_hf.transpose(1, 2), pos_hf, "bsnh"
        )  # output is BxSxNxH

        self.assertTrue(torch.allclose(x_bnsh_after_rope, x_bsnh_after_rope.transpose(1, 2)))

    # HF rotary == MSFT rotary non-interleaved
    def test_hf_rotary_and_msft_rotary_noninterleaved(self):
        x_bnsh = torch.randn(
            self.config.batch_size, self.config.num_heads, self.config.sequence_length, self.config.head_size
        )
        cos_hf, sin_hf = self.llama_hf.get_cos_sin_cache(self.config.sequence_length)
        pos_hf = torch.stack([torch.arange(0, self.config.sequence_length) for _ in range(self.config.batch_size)])
        output_hf = self.llama_hf(x_bnsh, cos_hf, sin_hf, pos_hf)  # output is BxNxSxH

        x_bsnh = x_bnsh.transpose(1, 2)
        x_bsd = deepcopy(x_bsnh)  # deepcopy to avoid changes made by self.llama_ms forward pass
        cos_ms, sin_ms = self.llama_ms.get_cos_sin_cache()
        pos_ms = 0
        output_ms = (
            self.llama_ms(x_bsd, cos_ms, sin_ms, pos_ms, interleaved=False).detach().cpu().numpy()  # output is BxSxNxH
        )

        # Compare caches as Mx(H/2)
        self.assertTrue(
            torch.allclose(self.llama_hf.cos_cached.squeeze()[:, : (self.config.head_size // 2)], cos_ms.squeeze())
        )
        self.assertTrue(
            torch.allclose(self.llama_hf.sin_cached.squeeze()[:, : (self.config.head_size // 2)], sin_ms.squeeze())
        )

        # Compare outputs as BxSxNxH
        self.assertTrue(np.allclose(output_hf.transpose(1, 2).detach().cpu().numpy(), output_ms))

    # Prompt step, interleaved = true, pos ids shape = (1)
    def test_msft_prompt_rotary_interleaved(self):
        # Calculated this way to match the data in rotary_embedding_op_test.cc
        x_bnsh = torch.randn(
            self.config.batch_size, self.config.num_heads, self.config.sequence_length, self.config.head_size
        )
        x_bsnh = x_bnsh.transpose(1, 2)
        x_bsd = deepcopy(x_bsnh)  # deepcopy to avoid changes made by self.llama_ms forward pass
        cos_ms, sin_ms = self.llama_ms.get_cos_sin_cache()
        pos_ms = 0
        output_ms = self.llama_ms(deepcopy(x_bsnh), cos_ms, sin_ms, pos_ms, interleaved=True).detach().cpu().numpy()

        x_bsd = x_bsd.reshape(self.config.batch_size, self.config.sequence_length, self.config.hidden_size)
        pos_ms = torch.tensor([pos_ms])
        onnx_graph = self.create_onnx_graph(x_bsd.shape, pos_ms.shape, cos_ms, sin_ms, interleaved=True)
        inputs_ort = {
            "input": x_bsd.detach().cpu().numpy(),
            "position_ids": pos_ms.detach().cpu().numpy(),
        }

        # Compare inputs/outputs as BxSxNxH
        self.assertTrue(np.allclose(x_bsnh.flatten(), x_bsd.flatten()))
        self.run_ort_ep_tests(onnx_graph, inputs_ort, output_ms)

    # Token generation step, interleaved = true, pos ids shape = (1)
    def test_msft_token_rotary_interleaved(self):
        # Calculated this way to match the data in rotary_embedding_op_test.cc
        x_bnsh = torch.randn(
            self.config.batch_size, self.config.num_heads, self.config.sequence_length, self.config.head_size
        )
        x_bsnh = x_bnsh.transpose(1, 2)
        x_bsd = deepcopy(x_bsnh)  # deepcopy to avoid changes made by self.llama_ms forward pass
        cos_ms, sin_ms = self.llama_ms.get_cos_sin_cache()
        pos_ms = 2
        output_ms = self.llama_ms(deepcopy(x_bsnh), cos_ms, sin_ms, pos_ms, interleaved=True).detach().cpu().numpy()

        x_bsd = x_bsd.reshape(self.config.batch_size, self.config.sequence_length, self.config.hidden_size)
        pos_ms = torch.tensor([pos_ms])
        onnx_graph = self.create_onnx_graph(x_bsd.shape, pos_ms.shape, cos_ms, sin_ms, interleaved=True)
        inputs_ort = {
            "input": x_bsd.detach().cpu().numpy(),
            "position_ids": pos_ms.detach().cpu().numpy(),
        }

        # Compare inputs/outputs as BxSxNxH
        self.assertTrue(np.allclose(x_bsnh.flatten(), x_bsd.flatten()))
        self.run_ort_ep_tests(onnx_graph, inputs_ort, output_ms)

    # Prompt step, interleaved = false, pos ids shape = (batch_size, sequence_length)
    def test_hf_prompt_rotary_batched_pos_ids(self):
        x_bnsh = torch.randn(
            self.config.batch_size, self.config.num_heads, self.config.sequence_length, self.config.head_size
        )
        cos_hf, sin_hf = self.llama_hf.get_cos_sin_cache(self.config.sequence_length)
        pos_ids = torch.stack([torch.arange(0, self.config.sequence_length) for _ in range(self.config.batch_size)])
        output_hf = self.llama_hf(x_bnsh, cos_hf, sin_hf, pos_ids)  # output is BxNxSxH

        x_bsnh = x_bnsh.transpose(1, 2)
        x_bsd = x_bsnh.reshape(self.config.batch_size, self.config.sequence_length, self.config.hidden_size)
        cos_ms, sin_ms = self.llama_ms.get_cos_sin_cache()
        onnx_graph = self.create_onnx_graph(x_bsd.shape, pos_ids.shape, cos_ms, sin_ms, interleaved=False)
        inputs_ort = {
            "input": x_bsd.detach().cpu().numpy(),
            "position_ids": pos_ids.detach().cpu().numpy(),
        }

        self.run_ort_ep_tests(onnx_graph, inputs_ort, output_hf.transpose(1, 2).detach().cpu().numpy())

    # Token generation step, interleaved = false, pos ids shape = (batch_size, sequence_length)
    def test_hf_token_rotary_batched_pos_ids(self):
        x_bnsh = torch.randn(self.config.batch_size, self.config.num_heads, 1, self.config.head_size)
        cos_hf, sin_hf = self.llama_hf.get_cos_sin_cache(self.config.sequence_length)
        pos_ids = torch.stack([torch.tensor([2]) for _ in range(self.config.batch_size)])
        output_hf = self.llama_hf(x_bnsh, cos_hf, sin_hf, pos_ids)  # output is BxNxSxH

        x_bsnh = x_bnsh.transpose(1, 2)
        x_bsd = x_bsnh.reshape(self.config.batch_size, 1, self.config.hidden_size)
        cos_ms, sin_ms = self.llama_ms.get_cos_sin_cache()
        onnx_graph = self.create_onnx_graph(x_bsd.shape, pos_ids.shape, cos_ms, sin_ms, interleaved=False)
        inputs_ort = {
            "input": x_bsd.detach().cpu().numpy(),
            "position_ids": pos_ids.detach().cpu().numpy(),
        }

        # Compare outputs as BxSxNxH
        self.run_ort_ep_tests(onnx_graph, inputs_ort, output_hf.transpose(1, 2).detach().cpu().numpy())

    # Bonus test: Prompt step, interleaved = false, pos ids shape = (1)
    def test_hf_prompt_rotary_one_pos_id(self):
        x_bnsh = torch.randn(
            self.config.batch_size, self.config.num_heads, self.config.sequence_length, self.config.head_size
        )
        cos_hf, sin_hf = self.llama_hf.get_cos_sin_cache(self.config.sequence_length)
        pos_hf = torch.stack([torch.arange(0, self.config.sequence_length) for _ in range(self.config.batch_size)])
        output_hf = self.llama_hf(x_bnsh, cos_hf, sin_hf, pos_hf)  # output is BxNxSxH

        x_bsnh = x_bnsh.transpose(1, 2)
        x_bsd = x_bsnh.reshape(self.config.batch_size, self.config.sequence_length, self.config.hidden_size)
        cos_ms, sin_ms = self.llama_ms.get_cos_sin_cache()
        pos_ms = torch.tensor([0])
        onnx_graph = self.create_onnx_graph(x_bsd.shape, pos_ms.shape, cos_ms, sin_ms, interleaved=False)
        inputs_ort = {
            "input": x_bsd.detach().cpu().numpy(),
            "position_ids": pos_ms.detach().cpu().numpy(),
        }

        # Compare outputs as BxSxNxH
        self.run_ort_ep_tests(onnx_graph, inputs_ort, output_hf.transpose(1, 2).detach().cpu().numpy())

    # Bonus test: Token generation step, interleaved = false, pos ids shape = (1)
    def test_hf_token_rotary_one_pos_id(self):
        x_bnsh = torch.randn(self.config.batch_size, self.config.num_heads, 1, self.config.head_size)
        cos_hf, sin_hf = self.llama_hf.get_cos_sin_cache(self.config.sequence_length)
        pos_ids = torch.stack([torch.tensor([2]) for _ in range(self.config.batch_size)])
        output_hf = self.llama_hf(x_bnsh, cos_hf, sin_hf, pos_ids)  # output is BxNxSxH

        x_bsnh = x_bnsh.transpose(1, 2)
        x_bsd = x_bsnh.reshape(self.config.batch_size, 1, self.config.hidden_size)
        cos_ms, sin_ms = self.llama_ms.get_cos_sin_cache()
        pos_ms = torch.tensor([2])
        onnx_graph = self.create_onnx_graph(x_bsd.shape, pos_ms.shape, cos_ms, sin_ms, interleaved=False)
        inputs_ort = {
            "input": x_bsd.detach().cpu().numpy(),
            "position_ids": pos_ms.detach().cpu().numpy(),
        }

        # Compare outputs as BxSxNxH
        self.run_ort_ep_tests(onnx_graph, inputs_ort, output_hf.transpose(1, 2).detach().cpu().numpy())


if __name__ == "__main__":
    unittest.main()
