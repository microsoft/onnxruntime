# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper

import onnxruntime as ort

torch = pytest.importorskip("torch")


def _run_case(dtype_name, mode):
    grouped = mode.startswith("grouped")
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    element_type = TensorProto.BFLOAT16 if dtype == torch.bfloat16 else TensorProto.FLOAT16
    rng = np.random.default_rng(33393)
    tokens, heads, kv_heads, width, past, block_size = 130, 24, 2, 256, 257, 16
    length = past + tokens
    blocks = (length + block_size - 1) // block_size
    selected_width = length + 3

    def floating(shape):
        return torch.tensor(rng.normal(0, 0.2, shape), dtype=dtype, device="cuda")

    query = floating((tokens, heads, width))
    key = floating((tokens, kv_heads, width))
    value = floating((tokens, kv_heads, width))
    key_cache = floating((blocks, block_size, kv_heads, width))
    value_cache = floating((blocks, block_size, kv_heads, width))
    # A suppressed write must read these deliberately different prefix-cache values.
    key_cache.reshape(-1, kv_heads, width)[past + 5] = 0.5
    value_cache.reshape(-1, kv_heads, width)[past + 5] = 5
    key[5] = -0.5
    value[5] = -5
    slots = np.arange(past, length, dtype=np.int32)
    slots[5] = -1
    block_table = np.arange(blocks, dtype=np.int32)
    block_table[2] = -1
    selected = np.tile(np.arange(selected_width, dtype=np.int32), (tokens, 1))
    selected[:, -3:] = [-1, length, length + 5]
    counts = np.full(tokens, selected_width, dtype=np.int32)
    counts[0] = 0
    counts[1] = 65
    counts[2] = 129
    selected[3] = -1
    sink = floating((heads,))
    sink[0] = 4.5

    q = query.float().cpu().numpy().astype(np.float64)
    k = key.float().cpu().numpy().astype(np.float64)
    v = value.float().cpu().numpy().astype(np.float64)
    kc = key_cache.float().cpu().numpy().reshape(-1, kv_heads, width).astype(np.float64)
    vc = value_cache.float().cpu().numpy().reshape(-1, kv_heads, width).astype(np.float64)
    sink_ref = sink.float().cpu().numpy().astype(np.float64)
    scale, softcap = 0.0625, 1.7
    expected = np.zeros((tokens, heads, width), dtype=np.float64)
    for token in range(tokens):
        positions = selected[token, : counts[token]]
        positions = positions[(positions >= 0) & (positions <= past + token) & (positions < length)]
        positions = positions[block_table[positions // block_size] >= 0]
        if positions.size == 0:
            continue
        keys, values = kc[positions].copy(), vc[positions].copy()
        for index, position in enumerate(positions):
            if position >= past and slots[position - past] >= 0:
                keys[index], values[index] = k[position - past], v[position - past]
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            logits = keys[:, kv_head] @ q[token, head] * scale
            logits = softcap * np.tanh(logits / softcap)
            maximum = max(logits.max(), sink_ref[head])
            weights = np.exp(logits - maximum)
            expected[token, head] = weights @ values[:, kv_head] / (weights.sum() + np.exp(sink_ref[head] - maximum))

    feed = {
        "query": query.reshape(tokens, heads * width),
        "key": key.reshape(tokens, kv_heads * width),
        "value": value.reshape(tokens, kv_heads * width),
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cumulative_sequence_length": torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
        "past_seqlens": torch.tensor([past], dtype=torch.int32, device="cuda"),
        "block_table": torch.tensor(block_table[None], device="cuda"),
        "slot_mapping": torch.tensor(slots, device="cuda"),
        "selected_indices": torch.tensor(selected, device="cuda"),
        "selected_counts": torch.tensor(counts, device="cuda"),
        "head_sink": sink,
    }
    names = [*list(feed)[:-1], "", "", "", "", "", "head_sink"]
    inputs = [
        helper.make_tensor_value_info(
            name, TensorProto.INT32 if tensor.dtype == torch.int32 else element_type, list(tensor.shape)
        )
        for name, tensor in feed.items()
    ]
    node = helper.make_node(
        "SparsePagedAttention",
        names,
        ["output"],
        domain="com.microsoft",
        num_heads=heads,
        kv_num_heads=kv_heads,
        scale=scale,
        softcap=softcap,
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "sparse_prefill_reference",
            inputs,
            [helper.make_tensor_value_info("output", element_type, [tokens, heads * width])],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    ort.set_default_logger_severity(0)
    ort.set_default_logger_verbosity(1)
    options = ort.SessionOptions()
    options.log_severity_level = 0
    options.log_verbosity_level = 1
    options.add_session_config_entry("ep.cuda.sparse_paged_attention_grouped", str(int(grouped)))
    options.add_session_config_entry(
        "ep.cuda.sparse_paged_attention_grouped_vectorized", str(int(mode == "grouped_vectorized"))
    )
    session = ort.InferenceSession(
        model.SerializeToString(),
        options,
        providers=[("CUDAExecutionProvider", {"device_id": torch.cuda.current_device()})],
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    binding = session.io_binding()
    device = torch.cuda.current_device()
    for name, tensor in feed.items():
        tensor_type = TensorProto.INT32 if tensor.dtype == torch.int32 else element_type
        binding.bind_input(name, "cuda", device, tensor_type, tuple(tensor.shape), tensor.data_ptr())
    output = torch.empty((tokens, heads * width), dtype=dtype, device="cuda")
    binding.bind_output("output", "cuda", device, element_type, tuple(output.shape), output.data_ptr())
    torch.cuda.synchronize()
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    actual = output.float().cpu().numpy().reshape(tokens, heads, width)
    tolerance = 0.004 if dtype == torch.float16 else 0.015
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    print("sparse prefill reference passed", flush=True)


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA EP required")
@pytest.mark.parametrize(
    "dtype,mode",
    [
        ("float16", "scalar"),
        ("float16", "qk"),
        ("float16", "qk_pv"),
        ("bfloat16", "qk"),
        ("float16", "grouped"),
        ("float16", "grouped_vectorized"),
    ],
)
def test_sparse_prefill_specializations_match_reference(dtype, mode):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SM80+ CUDA required")
    environment = dict(
        os.environ,
        ORT_SPARSE_PREFILL_TENSOR_CORE_QK=str(int(mode in ("qk", "qk_pv"))),
        ORT_SPARSE_PREFILL_TENSOR_CORE_PV=str(int(mode == "qk_pv")),
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--dtype", dtype, "--mode", mode],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected_dispatch = (
        f"SparsePagedAttention dispatch: grouped, vectorized={int(mode == 'grouped_vectorized')}"
        if mode.startswith("grouped")
        else f"SparsePagedAttention dispatch: tiled, heads=12, qk={int(mode != 'scalar')}, pv={int(mode == 'qk_pv')}"
    )
    assert expected_dispatch in result.stderr, result.stdout + result.stderr
    assert "sparse prefill reference passed" in result.stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], required=True)
    parser.add_argument("--mode", choices=["scalar", "qk", "qk_pv", "grouped", "grouped_vectorized"], required=True)
    args = parser.parse_args()
    _run_case(args.dtype, args.mode)
