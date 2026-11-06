# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import numpy as np
from onnx import TensorProto, helper

import onnxruntime as ort

GGUF_TYPES_INFO = [
    (0, 1, 4),
    (1, 1, 2),
    (2, 32, 18),
    (6, 32, 22),
    (8, 32, 34),
    (13, 256, 176),
    (14, 256, 210),
    (16, 256, 66),
    (17, 256, 74),
    (18, 256, 98),
    (20, 32, 18),
    (22, 256, 82),
    (23, 256, 136),
    (30, 1, 2),
    (42, 64, 18),
]


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    return (((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def _bf16_u16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


def _make_synthetic_gguf_rows(rng: np.random.Generator, rows: int, k: int, gtype: int, bel: int, bbytes: int) -> np.ndarray:
    row_bytes = (k // bel) * bbytes
    if gtype == 0:
        return (rng.standard_normal((rows, k), dtype=np.float32) * 0.1).view(np.uint8).reshape(rows, row_bytes)
    if gtype == 1:
        return (rng.standard_normal((rows, k)) * 0.1).astype(np.float16).view(np.uint8).reshape(rows, row_bytes)
    if gtype == 30:
        return _f32_to_bf16_u16(rng.standard_normal((rows, k)) * 0.1).view(np.uint8).reshape(rows, row_bytes)

    raw = rng.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
    for r in range(rows):
        for b in range(k // bel):
            off = b * bbytes + (208 if gtype == 14 else 0)
            raw[r, off] = 0x00
            raw[r, off + 1] = 0x20  # fp16 0.0078125
            if gtype == 13:
                raw[r, off + 2] = 0x00
                raw[r, off + 3] = 0x1C
    return raw


@unittest.skipIf("CUDAExecutionProvider" not in ort.get_available_providers(), "CUDAExecutionProvider is required")
class TestGgufPackedOps(unittest.TestCase):
    def test_gather_packed_cpu_vs_cuda_all_types(self):
        rng = np.random.default_rng(42)
        k = 256
        vocab = 16
        indices = np.arange(vocab, dtype=np.int64)
        for gtype, bel, bbytes in GGUF_TYPES_INFO:
            row_bytes = (k // bel) * bbytes
            w_raw = _make_synthetic_gguf_rows(rng, vocab, k, gtype, bel, bbytes)
            node = helper.make_node(
                "GatherPacked",
                ["W", "I"],
                ["Y"],
                domain="com.microsoft",
                K=k,
                packed_type=gtype,
                output_dtype=int(TensorProto.FLOAT),
            )
            graph = helper.make_graph(
                [node],
                "gather_packed_test",
                [
                    helper.make_tensor_value_info("W", TensorProto.UINT8, [vocab, row_bytes]),
                    helper.make_tensor_value_info("I", TensorProto.INT64, [vocab]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [vocab, k])],
            )
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)]
            )
            sess_cpu = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            sess_cuda = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
            y_cpu = sess_cpu.run(None, {"W": w_raw, "I": indices})[0]
            y_cuda = sess_cuda.run(None, {"W": w_raw, "I": indices})[0]
            np.testing.assert_allclose(y_cpu, y_cuda, rtol=0.0, atol=0.0, err_msg=f"Mismatch for ggml_type={gtype}")

    def test_matmul_packed_cuda_all_types(self):
        rng = np.random.default_rng(123)
        k = 256
        n = 32
        indices = np.arange(n, dtype=np.int64)
        for gtype, bel, bbytes in GGUF_TYPES_INFO:
            row_bytes = (k // bel) * bbytes
            w_raw = _make_synthetic_gguf_rows(rng, n, k, gtype, bel, bbytes)

            # Dequantize reference via CPU GatherPacked
            g_node = helper.make_node(
                "GatherPacked", ["W", "I"], ["W_f32"], domain="com.microsoft", K=k, packed_type=gtype, output_dtype=1
            )
            g_model = helper.make_model(
                helper.make_graph(
                    [g_node],
                    "deq",
                    [
                        helper.make_tensor_value_info("W", TensorProto.UINT8, [n, row_bytes]),
                        helper.make_tensor_value_info("I", TensorProto.INT64, [n]),
                    ],
                    [helper.make_tensor_value_info("W_f32", TensorProto.FLOAT, [n, k])],
                ),
                opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)],
            )
            w_f32 = ort.InferenceSession(g_model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
                None, {"W": w_raw, "I": indices}
            )[0]

            # Build MatMulPacked model with Cast(FLOAT -> BFLOAT16) -> MatMulPacked -> Cast(BFLOAT16 -> FLOAT)
            nodes = [
                helper.make_node("Cast", ["A_f32"], ["A_bf16"], to=int(TensorProto.BFLOAT16)),
                helper.make_node(
                    "MatMulPacked",
                    ["A_bf16", "B"],
                    ["Y_bf16"],
                    domain="com.microsoft",
                    K=k,
                    N=n,
                    packed_type=gtype,
                ),
                helper.make_node("Cast", ["Y_bf16"], ["Y_f32"], to=int(TensorProto.FLOAT)),
            ]
            mm_model = helper.make_model(
                helper.make_graph(
                    nodes,
                    "mm_packed",
                    [
                        helper.make_tensor_value_info("A_f32", TensorProto.FLOAT, ["M", k]),
                        helper.make_tensor_value_info("B", TensorProto.UINT8, [n, row_bytes]),
                    ],
                    [helper.make_tensor_value_info("Y_f32", TensorProto.FLOAT, ["M", n])],
                ),
                opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)],
            )
            w_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(w_f32))
            sess = ort.InferenceSession(mm_model.SerializeToString(), providers=["CUDAExecutionProvider"])
            for m in (1, 7, 8, 32):
                a_f32 = _bf16_u16_to_f32(_f32_to_bf16_u16(rng.standard_normal((m, k), dtype=np.float32) * 0.2))
                y_got = sess.run(None, {"A_f32": a_f32, "B": w_raw})[0]
                w_target = w_f32 if m <= 8 else w_bf16
                y_ref = _bf16_u16_to_f32(_f32_to_bf16_u16(a_f32 @ w_target.T))
                np.testing.assert_allclose(
                    y_got, y_ref, rtol=1e-2, atol=1e-2, err_msg=f"MatMulPacked mismatch for ggml_type={gtype}, M={m}"
                )

    def test_moe_packed_cuda_decode_and_prefill(self):
        rng = np.random.default_rng(999)
        num_experts = 16
        hidden_size = 256
        inter_size = 64
        top_k = 4
        gate_type, gate_bel, gate_bb = 16, 256, 66  # IQ2_XXS
        up_type, up_bel, up_bb = 16, 256, 66
        down_type, down_bel, down_bb = 42, 64, 18  # Q2_0

        gate_rb = (hidden_size // gate_bel) * gate_bb
        up_rb = (hidden_size // up_bel) * up_bb
        down_rb = (inter_size // down_bel) * down_bb

        gate_raw = _make_synthetic_gguf_rows(
            rng, num_experts * inter_size, hidden_size, gate_type, gate_bel, gate_bb
        ).reshape(num_experts, inter_size, gate_rb)
        up_raw = _make_synthetic_gguf_rows(
            rng, num_experts * inter_size, hidden_size, up_type, up_bel, up_bb
        ).reshape(num_experts, inter_size, up_rb)
        down_raw = _make_synthetic_gguf_rows(
            rng, num_experts * hidden_size, inter_size, down_type, down_bel, down_bb
        ).reshape(num_experts, hidden_size, down_rb)

        nodes = [
            helper.make_node("Cast", ["X_f32"], ["X_bf16"], to=int(TensorProto.BFLOAT16)),
            helper.make_node("Cast", ["R_f32"], ["R_bf16"], to=int(TensorProto.BFLOAT16)),
            helper.make_node(
                "MoEPacked",
                ["X_bf16", "R_bf16", "G", "U", "D"],
                ["Y_bf16"],
                domain="com.microsoft",
                k=top_k,
                normalize_routing_weights=1,
                activation_type="swiglu",
                gate_type=gate_type,
                up_type=up_type,
                down_type=down_type,
            ),
            helper.make_node("Cast", ["Y_bf16"], ["Y_f32"], to=int(TensorProto.FLOAT)),
        ]
        moe_model = helper.make_model(
            helper.make_graph(
                nodes,
                "moe_packed",
                [
                    helper.make_tensor_value_info("X_f32", TensorProto.FLOAT, ["M", hidden_size]),
                    helper.make_tensor_value_info("R_f32", TensorProto.FLOAT, ["M", num_experts]),
                    helper.make_tensor_value_info("G", TensorProto.UINT8, [num_experts, inter_size, gate_rb]),
                    helper.make_tensor_value_info("U", TensorProto.UINT8, [num_experts, inter_size, up_rb]),
                    helper.make_tensor_value_info("D", TensorProto.UINT8, [num_experts, hidden_size, down_rb]),
                ],
                [helper.make_tensor_value_info("Y_f32", TensorProto.FLOAT, ["M", hidden_size])],
            ),
            opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)],
        )
        sess = ort.InferenceSession(moe_model.SerializeToString(), providers=["CUDAExecutionProvider"])
        for m in (4, 48):
            x_f32 = rng.standard_normal((m, hidden_size), dtype=np.float32) * 0.2
            r_f32 = rng.standard_normal((m, num_experts), dtype=np.float32)
            y_got = sess.run(None, {"X_f32": x_f32, "R_f32": r_f32, "G": gate_raw, "U": up_raw, "D": down_raw})[0]
            self.assertEqual(y_got.shape, (m, hidden_size))
            self.assertTrue(np.all(np.isfinite(y_got)))


if __name__ == "__main__":
    unittest.main()
