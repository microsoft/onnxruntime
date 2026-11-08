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


def _round_to_dtype(arr, dtype):
    if dtype == TensorProto.BFLOAT16:
        return _bf16_u16_to_f32(_f32_to_bf16_u16(arr))
    return arr.astype(np.float16).astype(np.float32)


def _session(model, provider):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    if provider == "CUDAExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=[provider])
    session.disable_fallback()
    if provider not in session.get_providers():
        raise RuntimeError(f"Required provider {provider} was not initialized")
    return session


def _model(nodes, inputs, outputs):
    return helper.make_model(
        helper.make_graph(nodes, "packed_ops", inputs, outputs),
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)],
    )


def _gather_model(vocab, k, row_bytes, gtype, dtype=TensorProto.FLOAT, index_type=TensorProto.INT64):
    return _model(
        [
            helper.make_node(
                "GatherPacked",
                ["W", "I"],
                ["Y_typed"],
                domain="com.microsoft",
                K=k,
                packed_type=gtype,
                output_dtype=int(dtype),
            ),
            helper.make_node("Cast", ["Y_typed"], ["Y"], to=int(TensorProto.FLOAT)),
        ],
        [
            helper.make_tensor_value_info("W", TensorProto.UINT8, [vocab, row_bytes]),
            helper.make_tensor_value_info("I", index_type, ["rows"]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["rows", k])],
    )


def _dequantize_weights(raw, k, gtype):
    rows = raw.reshape(-1, raw.shape[-1])
    session = _session(_gather_model(rows.shape[0], k, rows.shape[1], gtype), "CPUExecutionProvider")
    result = session.run(None, {"W": rows, "I": np.arange(rows.shape[0], dtype=np.int64)})[0]
    return result.reshape(*raw.shape[:-1], k)


def _matmul_model(k, n, row_bytes, gtype, dtype=TensorProto.BFLOAT16):
    return _model(
        [
            helper.make_node("Cast", ["A_f32"], ["A_typed"], to=int(dtype)),
            helper.make_node(
                "MatMulPacked",
                ["A_typed", "B"],
                ["Y_typed"],
                domain="com.microsoft",
                K=k,
                N=n,
                packed_type=gtype,
            ),
            helper.make_node("Cast", ["Y_typed"], ["Y_f32"], to=int(TensorProto.FLOAT)),
        ],
        [
            helper.make_tensor_value_info("A_f32", TensorProto.FLOAT, ["M", k]),
            helper.make_tensor_value_info("B", TensorProto.UINT8, [n, row_bytes]),
        ],
        [helper.make_tensor_value_info("Y_f32", TensorProto.FLOAT, ["M", n])],
    )


def _moe_model(num_experts, hidden_size, inter_size, top_k, dtype, types, row_bytes, normalize):
    return _model(
        [
            helper.make_node("Cast", ["X_f32"], ["X"], to=int(dtype)),
            helper.make_node("Cast", ["R_f32"], ["R"], to=int(dtype)),
            helper.make_node(
                "MoEPacked",
                ["X", "R", "G", "U", "D"],
                ["Y"],
                domain="com.microsoft",
                k=top_k,
                normalize_routing_weights=int(normalize),
                activation_type="swiglu",
                gate_type=types[0],
                up_type=types[1],
                down_type=types[2],
            ),
            helper.make_node("Cast", ["Y"], ["Y_f32"], to=int(TensorProto.FLOAT)),
        ],
        [
            helper.make_tensor_value_info("X_f32", TensorProto.FLOAT, ["M", hidden_size]),
            helper.make_tensor_value_info("R_f32", TensorProto.FLOAT, ["M", num_experts]),
            helper.make_tensor_value_info("G", TensorProto.UINT8, [num_experts, inter_size, row_bytes[0]]),
            helper.make_tensor_value_info("U", TensorProto.UINT8, [num_experts, inter_size, row_bytes[1]]),
            helper.make_tensor_value_info("D", TensorProto.UINT8, [num_experts, hidden_size, row_bytes[2]]),
        ],
        [helper.make_tensor_value_info("Y_f32", TensorProto.FLOAT, ["M", hidden_size])],
    )


def _moe_reference(x, logits, gate, up, down, top_k, dtype, normalize):
    x = _round_to_dtype(x, dtype)
    logits = _round_to_dtype(logits, dtype)
    if dtype == TensorProto.BFLOAT16:
        gate, up, down = (_round_to_dtype(w, dtype) for w in (gate, up, down))
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    # Equal logits route to the lower expert index.
    experts = np.argsort(-logits, axis=1, kind="stable")[:, :top_k]
    scales = np.take_along_axis(probabilities, experts, axis=1)
    if normalize:
        scales /= scales.sum(axis=1, keepdims=True)
    result = np.zeros_like(x)
    for row in range(x.shape[0]):
        for rank, expert in enumerate(experts[row]):
            g = _round_to_dtype(x[row] @ gate[expert].T, dtype)
            u = _round_to_dtype(x[row] @ up[expert].T, dtype)
            activation = _round_to_dtype(u * (g / (1.0 + np.exp(-g))), dtype)
            expert_output = _round_to_dtype(activation @ down[expert].T, dtype)
            result[row] += scales[row, rank] * expert_output
    return _round_to_dtype(result, dtype)


def _make_synthetic_gguf_rows(
    rng: np.random.Generator, rows: int, k: int, gtype: int, bel: int, bbytes: int
) -> np.ndarray:
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


class TestGatherPackedValidation(unittest.TestCase):
    def test_empty_vocabulary(self):
        for provider in ("CPUExecutionProvider", "CUDAExecutionProvider"):
            if provider not in ort.get_available_providers():
                continue
            with self.subTest(provider=provider):
                session = _session(_gather_model(0, 32, 128, 0), provider)
                weights = np.empty((0, 128), dtype=np.uint8)
                empty = session.run(None, {"W": weights, "I": np.array([], dtype=np.int64)})[0]
                self.assertEqual(empty.shape, (0, 32))
                with self.assertRaisesRegex(Exception, "nonempty vocabulary"):
                    session.run(None, {"W": weights, "I": np.array([0], dtype=np.int64)})

    def test_reject_unsupported_formats(self):
        unsupported = [
            (3, 20),
            (7, 24),
            (9, 36),
            (10, 84),
            (11, 110),
            (12, 144),
            (15, 292),
            (19, 50),
            (21, 110),
            (29, 56),
            (39, 17),
            (40, 36),
            (41, 18),
            (1 << 32, 4),
        ]
        for provider in ("CPUExecutionProvider", "CUDAExecutionProvider"):
            if provider not in ort.get_available_providers():
                continue
            for gtype, row_bytes in unsupported:
                with (
                    self.subTest(provider=provider, gtype=gtype),
                    self.assertRaisesRegex(Exception, "unsupported packed_type"),
                ):
                    _session(_gather_model(1, 256, row_bytes, gtype), provider)


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
            sess_cpu = _session(model, "CPUExecutionProvider")
            sess_cuda = _session(model, "CUDAExecutionProvider")
            y_cpu = sess_cpu.run(None, {"W": w_raw, "I": indices})[0]
            y_cuda = sess_cuda.run(None, {"W": w_raw, "I": indices})[0]
            np.testing.assert_allclose(y_cpu, y_cuda, rtol=0.0, atol=0.0, err_msg=f"Mismatch for ggml_type={gtype}")

    def test_gather_packed_unquantized_short_rows(self):
        rng = np.random.default_rng(43)
        for gtype, bbytes in ((0, 4), (1, 2), (30, 2)):
            for k in (1, 8, 33):
                raw = _make_synthetic_gguf_rows(rng, 3, k, gtype, 1, bbytes)
                for dtype in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16):
                    for index_type, numpy_type in ((TensorProto.INT32, np.int32), (TensorProto.INT64, np.int64)):
                        with self.subTest(gtype=gtype, k=k, dtype=dtype, index_type=index_type):
                            model = _gather_model(3, k, k * bbytes, gtype, dtype, index_type)
                            inputs = {"W": raw, "I": np.array([2, 0, 1, -1, 3], dtype=numpy_type)}
                            cpu = _session(model, "CPUExecutionProvider").run(None, inputs)[0]
                            cuda = _session(model, "CUDAExecutionProvider").run(None, inputs)[0]
                            np.testing.assert_array_equal(cuda, cpu)

    def test_matmul_packed_cuda_all_types(self):
        rng = np.random.default_rng(123)
        k, n = 256, 32
        for gtype, bel, bbytes in GGUF_TYPES_INFO:
            raw = _make_synthetic_gguf_rows(rng, n, k, gtype, bel, bbytes)
            weights = _dequantize_weights(raw, k, gtype)
            session = _session(_matmul_model(k, n, raw.shape[1], gtype), "CUDAExecutionProvider")
            for m in (1, 7, 8, 32):
                with self.subTest(gtype=gtype, m=m):
                    a = _round_to_dtype(rng.standard_normal((m, k), dtype=np.float32) * 0.2, TensorProto.BFLOAT16)
                    got = session.run(None, {"A_f32": a, "B": raw})[0]
                    target = weights if m <= 8 else _round_to_dtype(weights, TensorProto.BFLOAT16)
                    ref = _round_to_dtype(a @ target.T, TensorProto.BFLOAT16)
                    np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)

    def test_matmul_packed_prefill_fp32_reduction(self):
        k, n = 4096, 32
        weights = np.ones((n, k), dtype=np.float32)
        weights[:, -1] = 1.125
        raw = weights.view(np.uint8).reshape(n, k * 4)
        for dtype in (TensorProto.BFLOAT16, TensorProto.FLOAT16):
            session = _session(_matmul_model(k, n, raw.shape[1], 0, dtype), "CUDAExecutionProvider")
            for m in (9, 32):
                with self.subTest(dtype=dtype, m=m):
                    a = np.ones((m, k), dtype=np.float32)
                    a[:, k // 2 :] = -1
                    got = session.run(None, {"A_f32": a, "B": raw})[0]
                    # Preserve the residual after cancellation of large partial sums.
                    np.testing.assert_array_equal(got, np.full((m, n), -0.125, dtype=np.float32))

    def test_matmul_packed_partial_tensor_core_tile(self):
        rng = np.random.default_rng(124)
        k, n = 32, 65537
        # F32 values must be rounded to BF16 before the Tensor Core multiplication.
        raw = _make_synthetic_gguf_rows(rng, n, k, 0, 1, 4)
        weights = _round_to_dtype(raw.view(np.float32).reshape(n, k), TensorProto.BFLOAT16)
        session = _session(_matmul_model(k, n, raw.shape[1], 0), "CUDAExecutionProvider")
        for m in (1, 7, 8):
            with self.subTest(m=m):
                a = _round_to_dtype(rng.standard_normal((m, k), dtype=np.float32), TensorProto.BFLOAT16)
                got = session.run(None, {"A_f32": a, "B": raw})[0]
                ref = _round_to_dtype(a @ weights.T, TensorProto.BFLOAT16)
                np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)

    def test_reject_unsupported_matmul_and_moe_formats(self):
        with self.assertRaisesRegex(Exception, "unsupported packed_type"):
            _session(_matmul_model(32, 8, 20, 3), "CUDAExecutionProvider")
        for index, name in enumerate(("gate", "up", "down")):
            with self.subTest(weight=name):
                types = [0, 0, 0]
                row_bytes = [128, 128, 128]
                types[index], row_bytes[index] = 3, 20
                model = _moe_model(2, 32, 32, 1, TensorProto.FLOAT16, types, row_bytes, True)
                with self.assertRaisesRegex(Exception, f"unsupported {name}_type"):
                    _session(model, "CUDAExecutionProvider")

    def test_moe_packed_reject_top_k_above_expert_count(self):
        model = _moe_model(2, 32, 32, 4, TensorProto.FLOAT16, (0, 0, 0), (128, 128, 128), True)
        session = _session(model, "CUDAExecutionProvider")
        inputs = {
            "X_f32": np.zeros((1, 32), dtype=np.float32),
            "R_f32": np.zeros((1, 2), dtype=np.float32),
            "G": np.zeros((2, 32, 128), dtype=np.uint8),
            "U": np.zeros((2, 32, 128), dtype=np.uint8),
            "D": np.zeros((2, 32, 128), dtype=np.uint8),
        }
        with self.assertRaisesRegex(Exception, "k <= num_experts"):
            session.run(None, inputs)

    def _check_moe(self, num_experts, hidden_size, inter_size, dtype, formats, row_counts, normalize):
        rng = np.random.default_rng(999)
        top_k = min(4, num_experts)
        raw_weights = []
        weights = []
        for out_size, k, (gtype, bel, bbytes) in zip(
            (inter_size, inter_size, hidden_size), (hidden_size, hidden_size, inter_size), formats, strict=True
        ):
            raw = _make_synthetic_gguf_rows(rng, num_experts * out_size, k, gtype, bel, bbytes)
            raw = raw.reshape(num_experts, out_size, -1)
            raw_weights.append(raw)
            weights.append(_dequantize_weights(raw, k, gtype))
        model = _moe_model(
            num_experts,
            hidden_size,
            inter_size,
            top_k,
            dtype,
            [f[0] for f in formats],
            [w.shape[-1] for w in raw_weights],
            normalize,
        )
        session = _session(model, "CUDAExecutionProvider")
        for m in row_counts:
            with self.subTest(
                experts=num_experts, hidden=hidden_size, inter=inter_size, dtype=dtype, m=m, normalize=normalize
            ):
                x = rng.standard_normal((m, hidden_size), dtype=np.float32) * 0.2
                logits = rng.standard_normal((m, num_experts), dtype=np.float32)
                # Concentrate routes so BF16 prefill fills multiple eight-route subtiles.
                logits[:, :8] += 4.0
                logits[0] = 0.0  # Exercise tie-breaking in both routing kernels.
                got = session.run(
                    None, {"X_f32": x, "R_f32": logits, "G": raw_weights[0], "U": raw_weights[1], "D": raw_weights[2]}
                )[0]
                ref = _moe_reference(x, logits, *weights, top_k, dtype, normalize)
                self.assertEqual(got.shape, (m, hidden_size))
                self.assertTrue(np.all(np.isfinite(got)))
                self.assertGreater(np.max(np.abs(ref)), 1e-2)
                np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-3)

    def test_moe_packed_cuda_decode_and_prefill(self):
        formats = ((16, 256, 66), (16, 256, 66), (42, 64, 18))
        for dtype in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
            for num_experts in (16, 512):
                for normalize in (False, True):
                    self._check_moe(num_experts, 256, 64, dtype, formats, (4, 48), normalize)

    def test_moe_packed_fp16_large_activations(self):
        formats = ((1, 1, 2),) * 3
        for hidden_size, inter_size in ((4096, 64), (256, 4096)):
            self._check_moe(2, hidden_size, inter_size, TensorProto.FLOAT16, formats, (4, 48), True)


if __name__ == "__main__":
    unittest.main()
