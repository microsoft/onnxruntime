# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
import os
import unittest

import numpy
import torch
from bert_padding import pad_input, unpad_input
from einops import rearrange, repeat
from onnx import TensorProto, helper
from parameterized import parameterized
from test_gqa import attention_ref, has_flash_attention

from onnxruntime import InferenceSession, SessionOptions

torch.manual_seed(0)

pipeline_mode = True  # Reduces number of tests so pipeline doesn't time out


class Formats:
    BSNH = 0
    BNSH = 1


class Config:
    batch_size = 0
    sequence_length = 0
    kv_sequence_length = 0  # this is past sequence length when there is past state.
    num_heads = 0
    kv_num_heads = 0
    head_size = 0
    ep = "CUDAExecutionProvider"

    def __init__(self, batch_size, sequence_length, kv_sequence_length, num_heads, kv_num_heads, head_size):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.kv_sequence_length = kv_sequence_length
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size

    def __repr__(self):
        short_ep = self.ep[: -len("ExecutionProvider")].lower()
        return (
            f"Config(batch_size={self.batch_size}, sequence_length={self.sequence_length}, "
            f"kv_sequence_length={self.kv_sequence_length}, "
            f"num_heads={self.num_heads}, kv_num_heads={self.kv_num_heads}, head_size={self.head_size}, ep={short_ep})"
        )


def create_packed_multihead_attention_graph(config: Config):
    nodes = [
        helper.make_node(
            "PackedMultiHeadAttention",
            [
                "query",
                "",
                "",
                "",
                "token_offset",
                "cumulative_sequence_length",
            ],
            ["output"],
            "PackedMultiHeadAttention_0",
            num_heads=config.num_heads,
            domain="com.microsoft",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "PackedMultiHeadAttention_Graph",
        [
            helper.make_tensor_value_info(
                "query",
                TensorProto.FLOAT16,
                [
                    -1,
                    config.num_heads,
                    3,
                    config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "token_offset", TensorProto.INT32, [config.batch_size, config.sequence_length]
            ),
            helper.make_tensor_value_info("cumulative_sequence_length", TensorProto.INT32, [config.batch_size + 1]),
        ],
        [
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT16,
                [-1, config.num_heads * config.head_size],
            ),
        ],
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_multihead_attention_graph(config: Config):
    nodes = [
        helper.make_node(
            "MultiHeadAttention",
            [
                "query",
                "key",
                "value",
            ],
            ["output"],
            "MultiHeadAttention_0",
            num_heads=config.num_heads,
            domain="com.microsoft",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "MultiHeadAttention_Graph",
        [
            helper.make_tensor_value_info(
                "query",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "key",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
        ],
        [
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT16,
                [config.batch_size, config.sequence_length, config.num_heads * config.head_size],
            ),
        ],
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen, (batch_size, 1), device=device)
    else:
        lengths = torch.randint(max_seqlen // 3, max_seqlen, (batch_size, 1), device=device)
    padding_mask = repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    return padding_mask


def generate_packed_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)

        def output_pad_fn(output_unpad):
            return pad_input(output_unpad, indices_q, batch_size, seqlen_q)

    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q

        def output_pad_fn(output_unpad):
            return rearrange(output_unpad, "(b s) h d -> b s h d", b=batch_size)

    if key_padding_mask is not None:
        k_unpad, _, _, _ = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")

    assert (query_padding_mask == key_padding_mask).all()
    assert nheads == nheads_k
    qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
    qkv = torch.stack([q, k, v], dim=2)
    if query_padding_mask is not None:

        def dqkv_pad_fn(dqkv_unpad):
            return pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)

    else:

        def dqkv_pad_fn(dqkv_unpad):
            return rearrange(dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)

    return (
        qkv_unpad.detach().requires_grad_(),
        cu_seqlens_q,
        max_seqlen_q,
        qkv.detach().requires_grad_(),
        output_pad_fn,
        dqkv_pad_fn,
    )


def create_inputs(config: Config):
    qkv = torch.randn(
        config.batch_size,
        config.sequence_length,
        3,
        config.num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    padding_mask = generate_random_padding_mask(config.sequence_length, config.batch_size, device="cuda", mode="random")
    qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn = generate_packed_qkv(
        *qkv.unbind(dim=2), padding_mask, padding_mask
    )
    return qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn, padding_mask


def generate_token_offset(cu_seqlens, max_seqlen):
    token_offset = []
    token_padset = []  # These are the indices that contain padding tokens
    for i in range(1, len(cu_seqlens)):
        start = i - 1
        pre_seqlen = cu_seqlens[i - 1]
        seqlen = cu_seqlens[i]
        token_offset += range(start * max_seqlen, (start * max_seqlen) + (seqlen - pre_seqlen))
        token_padset += range((start * max_seqlen) + (seqlen - pre_seqlen), i * max_seqlen)
    return numpy.asarray(token_offset + token_padset, dtype=numpy.int32)


def flash_attn_varlen_qkvpacked_func(qkv_unpad, cu_seqlens, token_offset, config):
    onnx_model_str = create_packed_multihead_attention_graph(config)
    qkv_unpad = torch.swapdims(qkv_unpad, 1, 2)
    ort_inputs = {
        "query": qkv_unpad.detach().cpu().numpy(),
        "token_offset": token_offset,
        "cumulative_sequence_length": cu_seqlens.cpu().numpy(),
    }
    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[config.ep])
    ort_output = ort_session.run(None, ort_inputs)
    output = torch.tensor(ort_output)
    return output


def mha_func(q, k, v, config):
    onnx_model_str = create_multihead_attention_graph(config)
    q = torch.reshape(q, (config.batch_size, config.sequence_length, -1))
    k = torch.reshape(k, (config.batch_size, config.kv_sequence_length, -1))
    v = torch.reshape(v, (config.batch_size, config.kv_sequence_length, -1))
    ort_inputs = {
        "query": q.detach().cpu().numpy(),
        "key": k.detach().cpu().numpy(),
        "value": v.detach().cpu().numpy(),
    }
    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[config.ep])
    ort_output = ort_session.run(None, ort_inputs)
    ort_output = numpy.array(ort_output)
    output = torch.tensor(ort_output)
    return output


def attention_qkvpacked_ref(
    qkv,
    key_padding_mask=None,
    causal=False,
    use_smooth_softmax=False,
):
    return attention_ref(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        query_padding_mask=key_padding_mask,
        key_padding_mask=key_padding_mask,
        causal=causal,
        use_smooth_softmax=use_smooth_softmax,
    )


def parity_check_mha(
    config,
    packed,
    rtol=1e-3,
    atol=1e-3,
):
    if packed:
        qkv_unpad, cu_seqlens, _, qkv, output_pad_fn, _, key_padding_mask = create_inputs(config)
        token_offset = generate_token_offset(cu_seqlens, config.sequence_length).reshape(
            (config.batch_size, config.sequence_length)
        )
        # ORT Flash
        out_unpad = flash_attn_varlen_qkvpacked_func(qkv_unpad, cu_seqlens, token_offset, config)
        out_unpad = torch.squeeze(out_unpad, 0)
        out = torch.reshape(
            output_pad_fn(out_unpad), (config.batch_size, config.sequence_length, config.num_heads, config.head_size)
        )
        out = out.detach().cpu().numpy()
        # Pytorch to compare
        out_ref, _ = attention_qkvpacked_ref(qkv, key_padding_mask, causal=False)
        out_ref = out_ref.detach().cpu().numpy()
    else:
        q = torch.randn(
            config.batch_size,
            config.sequence_length,
            config.num_heads,
            config.head_size,
            device="cuda",
            dtype=torch.float16,
            requires_grad=False,
        )
        k = torch.randn(
            config.batch_size,
            config.kv_sequence_length,
            config.kv_num_heads,
            config.head_size,
            device="cuda",
            dtype=torch.float16,
            requires_grad=False,
        )
        v = torch.randn(
            config.batch_size,
            config.kv_sequence_length,
            config.kv_num_heads,
            config.head_size,
            device="cuda",
            dtype=torch.float16,
            requires_grad=False,
        )
        out = mha_func(q, k, v, config)
        out = torch.squeeze(out, 0)
        out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))
        out = out.detach().cpu().numpy()
        # Pytorch to compare
        out_ref, _ = attention_ref(q, k, v, None, None, 0.0, None, causal=False)
        out_ref = out_ref.detach().cpu().numpy()

    numpy.testing.assert_allclose(
        out, out_ref, rtol=rtol, atol=atol, equal_nan=True, err_msg=f" with {config} packed={packed}"
    )


def packed_mha_test_cases():
    batch_sizes = [2] if pipeline_mode else [1, 5]
    sequence_lengths = [1024, 1025] if pipeline_mode else [1024, 1025, 2048]
    num_heads = [1, 3] if pipeline_mode else [1, 6, 16]
    head_sizes = [16, 256] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]

    for b in batch_sizes:
        for s in sequence_lengths:
            for n in num_heads:
                for h in head_sizes:
                    config = Config(b, s, s, n, n, h)
                    yield str(config), config


def mha_test_cases():
    batch_sizes = [2] if pipeline_mode else [1, 5]
    sequence_lengths = (
        [(1, 128), (113, 211), (2048, 2048)]
        if pipeline_mode
        else [
            (113, 203),
            (128, 217),
            (113, 211),
            (108, 256),
            (256, 512),
            (512, 256),
            (1024, 1024),
            (1023, 1024),
            (1024, 1023),
            (2048, 2048),
        ]
    )
    num_heads = [3] if pipeline_mode else [1, 6, 16]
    head_sizes = [64] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]

    for b in batch_sizes:
        for s, kv_sequence_length in sequence_lengths:
            for n in num_heads:
                for h in head_sizes:
                    config = Config(b, s, kv_sequence_length, n, n, h)
                    yield str(config), config


class TestMHA(unittest.TestCase):
    @parameterized.expand(packed_mha_test_cases())
    def test_packed_mha(self, _, config):
        if not has_flash_attention():
            return
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        print("-------- TEST PACKED MHA ---------")
        parity_check_mha(config, True)

    @parameterized.expand(mha_test_cases())
    def test_mha(self, _, config):
        if not has_flash_attention():
            return
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        print("-------- TEST MHA ---------")
        parity_check_mha(config, False)


if __name__ == "__main__":
    unittest.main()
