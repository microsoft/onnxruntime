# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

import onnxruntime


onnxruntime.preload_dlls()


def _has_cuda_ep() -> bool:
    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


def _run_onnx(model_bytes: bytes, inputs: dict[str, np.ndarray], provider: str) -> list[np.ndarray]:
    session = onnxruntime.InferenceSession(model_bytes, providers=[provider])
    return session.run(None, inputs)


def _torch_linear_attention_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    past_state: np.ndarray,
    decay: np.ndarray,
    beta: np.ndarray,
    q_num_heads: int,
    kv_num_heads: int,
    d_k: int,
    d_v: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    q = torch.from_numpy(query)
    k = torch.from_numpy(key)
    v = torch.from_numpy(value)
    s = torch.from_numpy(past_state).clone()
    g = torch.from_numpy(decay)
    b = torch.from_numpy(beta)

    batch, seq_len, _ = query.shape
    heads_per_group = q_num_heads // kv_num_heads
    output = torch.empty((batch, seq_len, q_num_heads * d_v), dtype=torch.float32)

    for bi in range(batch):
        for hk in range(kv_num_heads):
            state = s[bi, hk]
            for t in range(seq_len):
                kt = k[bi, t, hk * d_k : (hk + 1) * d_k]
                vt = v[bi, t, hk * d_v : (hk + 1) * d_v]
                exp_g = torch.exp(g[bi, t, hk])
                state = state * exp_g
                retrieved = torch.matmul(kt, state)
                bt = b[bi, t, 0]
                delta = bt * (vt - retrieved)
                state = state + torch.outer(kt, delta)
                for hg in range(heads_per_group):
                    hq = hk * heads_per_group + hg
                    qt = q[bi, t, hq * d_k : (hq + 1) * d_k]
                    ot = scale * torch.matmul(qt, state)
                    output[bi, t, hq * d_v : (hq + 1) * d_v] = ot
            s[bi, hk] = state

    return output.numpy(), s.numpy()


def _torch_causal_conv_reference(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    past_state: np.ndarray,
    activation: str,
) -> tuple[np.ndarray, np.ndarray]:
    xt = torch.from_numpy(x)
    wt = torch.from_numpy(weight)
    bt = torch.from_numpy(bias)
    pst = torch.from_numpy(past_state)

    pad = wt.shape[2] - 1
    padded = torch.cat([pst, xt], dim=2)
    out = torch.nn.functional.conv1d(padded, wt, bias=bt, stride=1, padding=0, groups=xt.shape[1])

    if activation in ("silu", "swish"):
        out = torch.nn.functional.silu(out)

    present = padded[:, :, -pad:] if pad > 0 else torch.empty((xt.shape[0], xt.shape[1], 0), dtype=torch.float32)
    return out.numpy(), present.numpy()


def _build_linear_attention_model(
    q_num_heads: int,
    kv_num_heads: int,
    update_rule: str,
    scale: float,
) -> bytes:
    node = helper.make_node(
        "LinearAttention",
        ["query", "key", "value", "past_state", "decay", "beta"],
        ["output", "present_state"],
        domain="com.microsoft",
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        update_rule=update_rule,
        scale=scale,
    )

    graph = helper.make_graph(
        [node],
        "LinearAttentionParity",
        [
            helper.make_tensor_value_info("query", TensorProto.FLOAT, ["B", "T", "QH"]),
            helper.make_tensor_value_info("key", TensorProto.FLOAT, ["B", "T", "KH"]),
            helper.make_tensor_value_info("value", TensorProto.FLOAT, ["B", "T", "VH"]),
            helper.make_tensor_value_info("past_state", TensorProto.FLOAT, ["B", "H", "DK", "DV"]),
            helper.make_tensor_value_info("decay", TensorProto.FLOAT, ["B", "T", "H"]),
            helper.make_tensor_value_info("beta", TensorProto.FLOAT, ["B", "T", 1]),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, ["B", "T", "OH"]),
            helper.make_tensor_value_info("present_state", TensorProto.FLOAT, ["B", "H", "DK", "DV"]),
        ],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _build_causal_conv_model(activation: str) -> bytes:
    node = helper.make_node(
        "CausalConvWithState",
        ["input", "weight", "bias", "past_state"],
        ["output", "present_state"],
        domain="com.microsoft",
        ndim=1,
        activation=activation,
    )

    graph = helper.make_graph(
        [node],
        "CausalConvWithStateParity",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, ["B", "C", "L"]),
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, ["C", 1, "K"]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, ["C"]),
            helper.make_tensor_value_info("past_state", TensorProto.FLOAT, ["B", "C", "P"]),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, ["B", "C", "L"]),
            helper.make_tensor_value_info("present_state", TensorProto.FLOAT, ["B", "C", "P"]),
        ],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


@unittest.skipUnless(_has_cuda_ep(), "CUDAExecutionProvider is required for parity tests")
class TestLinearAttentionCausalConvPyTorchParity(unittest.TestCase):
    def test_linear_attention_pytorch_parity(self):
        rng = np.random.default_rng(0)

        q_num_heads = 4
        kv_num_heads = 2
        d_k = 64
        d_v = 64
        batch = 2

        model = _build_linear_attention_model(
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            update_rule="gated_delta",
            scale=1.0 / np.sqrt(float(d_k)),
        )

        for seq_len in (1, 5):
            with self.subTest(seq_len=seq_len):
                inputs = {
                    "query": rng.standard_normal((batch, seq_len, q_num_heads * d_k), dtype=np.float32),
                    "key": rng.standard_normal((batch, seq_len, kv_num_heads * d_k), dtype=np.float32),
                    "value": rng.standard_normal((batch, seq_len, kv_num_heads * d_v), dtype=np.float32),
                    "past_state": rng.standard_normal((batch, kv_num_heads, d_k, d_v), dtype=np.float32),
                    "decay": rng.standard_normal((batch, seq_len, kv_num_heads), dtype=np.float32),
                    "beta": rng.uniform(0.0, 1.0, size=(batch, seq_len, 1)).astype(np.float32),
                }

                cuda_output, cuda_state = _run_onnx(model, inputs, "CUDAExecutionProvider")
                ref_output, ref_state = _torch_linear_attention_reference(
                    query=inputs["query"],
                    key=inputs["key"],
                    value=inputs["value"],
                    past_state=inputs["past_state"],
                    decay=inputs["decay"],
                    beta=inputs["beta"],
                    q_num_heads=q_num_heads,
                    kv_num_heads=kv_num_heads,
                    d_k=d_k,
                    d_v=d_v,
                    scale=1.0 / np.sqrt(float(d_k)),
                )

                output_max_diff = np.max(np.abs(cuda_output - ref_output))
                state_max_diff = np.max(np.abs(cuda_state - ref_state))
                print(
                    "LinearAttention parity "
                    f"(seq_len={seq_len}): output_max_diff={output_max_diff:.6e}, "
                    f"state_max_diff={state_max_diff:.6e}"
                )

                np.testing.assert_allclose(cuda_output, ref_output, rtol=2e-4, atol=3e-4)
                np.testing.assert_allclose(cuda_state, ref_state, rtol=2e-4, atol=3e-4)

    def test_causal_conv_with_state_pytorch_parity(self):
        rng = np.random.default_rng(1)

        batch = 2
        channels = 16
        kernel = 4
        pad = kernel - 1

        model = _build_causal_conv_model(activation="silu")

        for seq_len in (1, 7):
            with self.subTest(seq_len=seq_len):
                inputs = {
                    "input": rng.standard_normal((batch, channels, seq_len), dtype=np.float32),
                    "weight": rng.standard_normal((channels, 1, kernel), dtype=np.float32),
                    "bias": rng.standard_normal((channels,), dtype=np.float32),
                    "past_state": rng.standard_normal((batch, channels, pad), dtype=np.float32),
                }

                cuda_output, cuda_state = _run_onnx(model, inputs, "CUDAExecutionProvider")
                ref_output, ref_state = _torch_causal_conv_reference(
                    x=inputs["input"],
                    weight=inputs["weight"],
                    bias=inputs["bias"],
                    past_state=inputs["past_state"],
                    activation="silu",
                )

                output_max_diff = np.max(np.abs(cuda_output - ref_output))
                state_max_diff = np.max(np.abs(cuda_state - ref_state))
                print(
                    "CausalConvWithState parity "
                    f"(seq_len={seq_len}): output_max_diff={output_max_diff:.6e}, "
                    f"state_max_diff={state_max_diff:.6e}"
                )

                np.testing.assert_allclose(cuda_output, ref_output, rtol=1e-4, atol=2e-4)
                np.testing.assert_allclose(cuda_state, ref_state, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()