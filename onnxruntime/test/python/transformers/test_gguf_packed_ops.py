# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ctypes
import sys
import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper

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


def _cuda_bf16_supported():
    driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")

    def check(status):
        if status != 0:
            raise RuntimeError(f"Cannot query CUDA compute capability (CUDA driver error {status})")

    device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    check(driver.cuInit(0))
    check(driver.cuDeviceGet(ctypes.byref(device), 0))
    check(driver.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device))
    return major.value >= 8


def _activation_storage(values, dtype):
    if dtype == TensorProto.BFLOAT16:
        return _f32_to_bf16_u16(values)
    return np.ascontiguousarray(values, dtype=np.float16)


def _run_cuda_bound(session, inputs, output_shape, output_dtype, output_offset=2):
    binding = session.io_binding()
    owners = []
    for name, (dtype, values, offset) in inputs.items():
        storage = np.concatenate(
            (np.zeros(offset, dtype=np.uint8), np.ascontiguousarray(values).view(np.uint8).ravel())
        )
        owner = ort.OrtValue.ortvalue_from_numpy(storage, "cuda", 0)
        owners.append(owner)
        binding.bind_input(name, "cuda", 0, dtype, values.shape, owner.data_ptr() + offset)
    item_size = 4 if output_dtype == TensorProto.FLOAT else 2
    output_storage = ort.OrtValue.ortvalue_from_numpy(
        np.zeros(output_offset + int(np.prod(output_shape)) * item_size, dtype=np.uint8), "cuda", 0
    )
    binding.bind_output("Y", "cuda", 0, output_dtype, output_shape, output_storage.data_ptr() + output_offset)
    binding.synchronize_inputs()
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    raw = output_storage.numpy()[output_offset:]
    if output_dtype == TensorProto.BFLOAT16:
        return _bf16_u16_to_f32(raw.view(np.uint16)).reshape(output_shape)
    return (
        raw.view(np.float32 if output_dtype == TensorProto.FLOAT else np.float16)
        .reshape(output_shape)
        .astype(np.float32)
    )


def _session(model, provider, allow_unrounded_gemv=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    if provider == "CUDAExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if allow_unrounded_gemv is not None:
        options.add_session_config_entry("ep.cuda.matmul_packed_allow_unrounded_gemv", allow_unrounded_gemv)
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


def _moe_reference(x, logits, gate, up, down, top_k, dtype, normalize, round_weights=None):
    # BF16 CUDA kernels round dequantized weights to BF16 for the MMA; the CPU cold-expert kernels keep them in FP32.
    if round_weights is None:
        round_weights = dtype == TensorProto.BFLOAT16
    x = _round_to_dtype(x, dtype)
    logits = _round_to_dtype(logits, dtype)
    if round_weights:
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
    rng: np.random.Generator, rows: int, k: int, gtype: int, bel: int, bbytes: int, quant_scale: float = 2**-7
) -> np.ndarray:
    row_bytes = (k // bel) * bbytes
    if gtype == 0:
        return (rng.standard_normal((rows, k), dtype=np.float32) * 0.1).view(np.uint8).reshape(rows, row_bytes)
    if gtype == 1:
        return (rng.standard_normal((rows, k)) * 0.1).astype(np.float16).view(np.uint8).reshape(rows, row_bytes)
    if gtype == 30:
        return _f32_to_bf16_u16(rng.standard_normal((rows, k)) * 0.1).view(np.uint8).reshape(rows, row_bytes)

    raw = rng.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
    scale_bytes = np.array([quant_scale], dtype="<f2").view(np.uint8)
    min_bytes = np.array([quant_scale * 0.5], dtype="<f2").view(np.uint8)
    for r in range(rows):
        for b in range(k // bel):
            off = b * bbytes + (208 if gtype == 14 else 0)
            raw[r, off : off + 2] = scale_bytes
            if gtype == 13:
                raw[r, off + 2 : off + 4] = min_bytes
    return raw


class TestGatherPackedKnownAnswers(unittest.TestCase):
    def test_iq_high_codebook_indices_signs_and_scales(self):
        # Independent gguf-py answers from llama.cpp 970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc.
        cases = [
            (
                16,
                bytes.fromhex("0038") + bytes.fromhex("fffec08055d53f90") * 8,
                [
                    -9.5,
                    29.6875,
                    -9.5,
                    9.5,
                    -29.6875,
                    51.0625,
                    -51.0625,
                    51.0625,
                    9.5,
                    -9.5,
                    29.6875,
                    -29.6875,
                    9.5,
                    -51.0625,
                    51.0625,
                    -51.0625,
                    -29.6875,
                    -51.0625,
                    -29.6875,
                    -9.5,
                    -9.5,
                    -51.0625,
                    -29.6875,
                    -29.6875,
                    -9.5,
                    51.0625,
                    9.5,
                    9.5,
                    9.5,
                    51.0625,
                    51.0625,
                    -9.5,
                ],
            ),
            (
                17,
                bytes.fromhex("0038") + bytes.fromhex("ffabfe5580ff0003") * 8 + bytes([0xB6]) * 8,
                [
                    -34.9375,
                    34.9375,
                    -34.9375,
                    34.9375,
                    -34.9375,
                    34.9375,
                    -34.9375,
                    34.9375,
                    20.3125,
                    -6.5,
                    20.3125,
                    -34.9375,
                    34.9375,
                    -34.9375,
                    34.9375,
                    -34.9375,
                    -11.5,
                    -11.5,
                    -11.5,
                    -11.5,
                    -35.9375,
                    -11.5,
                    -61.8125,
                    -35.9375,
                    -61.8125,
                    35.9375,
                    11.5,
                    11.5,
                    11.5,
                    11.5,
                    11.5,
                    -35.9375,
                ],
            ),
            (
                22,
                bytes.fromhex("0038")
                + bytes.fromhex("fffe0000") * 8
                + bytes.fromhex("a55aff81") * 8
                + bytes([0xBF]) * 8
                + bytes([0xB6]) * 8,
                [
                    -34.9375,
                    34.9375,
                    -34.9375,
                    34.9375,
                    34.9375,
                    -34.9375,
                    34.9375,
                    -34.9375,
                    6.5,
                    -34.9375,
                    6.5,
                    -34.9375,
                    -34.9375,
                    34.9375,
                    -34.9375,
                    34.9375,
                    -11.5,
                    -61.8125,
                    -11.5,
                    -35.9375,
                    -11.5,
                    -11.5,
                    -61.8125,
                    -35.9375,
                    -35.9375,
                    35.9375,
                    61.8125,
                    11.5,
                    35.9375,
                    11.5,
                    11.5,
                    -35.9375,
                ],
            ),
            (
                18,
                bytes.fromhex("0038") + bytes.fromhex("fffec080fdbf817f") * 8 + bytes.fromhex("55d53fb0") * 8,
                [
                    -11.5,
                    80.5,
                    -149.5,
                    178.25,
                    -103.5,
                    57.5,
                    -126.5,
                    178.25,
                    80.5,
                    -103.5,
                    126.5,
                    -103.5,
                    11.5,
                    -178.25,
                    80.5,
                    -57.5,
                    -57.5,
                    -11.5,
                    -126.5,
                    -178.25,
                    -126.5,
                    -57.5,
                    -126.5,
                    -103.5,
                    -103.5,
                    178.25,
                    80.5,
                    57.5,
                    103.5,
                    34.5,
                    80.5,
                    -57.5,
                ],
            ),
        ]
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        for gtype, block, values in cases:
            raw = np.frombuffer(block, dtype=np.uint8).reshape(1, -1)
            expected = np.tile(np.array(values, dtype=np.float32), 8).reshape(1, -1)
            for provider in providers:
                for dtype in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16):
                    with self.subTest(gtype=gtype, provider=provider, dtype=dtype):
                        model = _gather_model(1, expected.shape[1], len(block), gtype, dtype)
                        got = _session(model, provider).run(None, {"W": raw, "I": np.array([0], np.int64)})[0]
                        ref = expected if dtype == TensorProto.FLOAT else _round_to_dtype(expected, dtype)
                        np.testing.assert_array_equal(got, ref)

    def test_ggml_known_answer_blocks(self):
        # Verified with gguf-py from llama.cpp commit 970b51ce7c9b, independently of the ORT decoder.
        nibbles = bytes.fromhex("f0e1d2c3b4a5968778695a4b3c2d1e0f")
        iq4_values = [-63.5, -52, -41.5, -32.5, -24.5, -17.5, -11, -5, 0.5, 6.5, 12.5, 19, 26.5, 34.5, 44.5, 56.5]
        iq4_values += iq4_values[::-1]
        cases = [
            (
                13,
                bytes.fromhex("003800344182c3044586c70891a2b3c4") + bytes([255]) * 32 + bytes([240]) * 128,
                np.repeat([6.75, 29.5, 22.25, 60, 129.75, 516.5, 393.25, 59], 32),
            ),
            (
                14,
                bytes([240]) * 128
                + bytes([228]) * 64
                + np.arange(-8, 8, dtype=np.int8).tobytes()
                + bytes.fromhex("0038"),
                np.repeat([128, 112, 48, 40, -30, -22.5, -31, -15.5, 0, -16, -16, -24, 30, 37.5, 93, 108.5], 16),
            ),
            (20, bytes.fromhex("0038") + nibbles, iq4_values),
            (
                23,
                bytes.fromhex("0038e4e410325476") + nibbles * 8,
                np.concatenate([np.array(iq4_values) * scale for scale in (-32, -15, 2, 19, -28, -11, 6, 23)]),
            ),
            (42, bytes.fromhex("0038") + bytes([228]) * 16, [-0.5, 0, 0.5, 1] * 16),
        ]
        for gtype, block, expected_values in cases:
            raw = np.frombuffer(block, dtype=np.uint8).reshape(1, -1)
            expected = np.array([expected_values], dtype=np.float32)
            for dtype in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16):
                with self.subTest(gtype=gtype, dtype=dtype):
                    model = _gather_model(1, expected.shape[1], len(block), gtype, dtype)
                    got = _session(model, "CPUExecutionProvider").run(None, {"W": raw, "I": np.array([0], np.int64)})[0]
                    ref = expected if dtype == TensorProto.FLOAT else _round_to_dtype(expected, dtype)
                    np.testing.assert_array_equal(got, ref)

    def test_little_endian_blocks(self):
        values = [1.0, -2.0, 0.5, 0.0]
        iq2_values = [5.375] + [1.0] * 31
        cases = [
            (0, bytes.fromhex("0000803f000000c00000003f00000000"), values),
            (1, bytes.fromhex("003c00c000380000"), values),
            (30, bytes.fromhex("803f00c0003f0000"), values),
            (
                2,
                bytes.fromhex("0038f0e1d2c3b4a5968778695a4b3c2d1e0f"),
                [i * 0.5 for i in range(-8, 8)] + [i * 0.5 for i in range(7, -9, -1)],
            ),
            (6, bytes.fromhex("003801000080") + bytes(16), [0.0] + [-8.0] * 30 + [0.0]),
            (
                8,
                bytes.fromhex("0038f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff000102030405060708090a0b0c0d0e0f"),
                [i * 0.5 for i in range(-16, 16)],
            ),
            (16, bytes.fromhex("003c") + bytes.fromhex("0100000000000000") * 8, iq2_values * 8),
            (17, bytes.fromhex("003c") + bytes.fromhex("0100000000000000") * 8 + bytes(8), iq2_values * 8),
            (22, bytes.fromhex("003c") + bytes.fromhex("01000000") * 8 + bytes(48), iq2_values * 8),
            (
                18,
                bytes.fromhex("003c") + bytes.fromhex("0100000000000000") * 8 + bytes(32),
                ([5.0] + [1.0] * 31) * 8,
            ),
        ]
        for gtype, block, expected in cases:
            raw = np.frombuffer(block, dtype=np.uint8).reshape(1, -1)
            for dtype in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16):
                with self.subTest(gtype=gtype, dtype=dtype):
                    model = _gather_model(1, len(expected), len(block), gtype, dtype)
                    got = _session(model, "CPUExecutionProvider").run(None, {"W": raw, "I": np.array([0], np.int64)})[0]
                    np.testing.assert_array_equal(got, np.array([expected], dtype=np.float32))

    def test_invalid_indices_select_first_row(self):
        raw = np.frombuffer(bytes.fromhex("0000803f0000004000004040"), dtype=np.uint8).reshape(3, 4)
        for index_type, numpy_type in ((TensorProto.INT32, np.int32), (TensorProto.INT64, np.int64)):
            with self.subTest(index_type=index_type):
                model = _gather_model(3, 1, 4, 0, index_type=index_type)
                got = _session(model, "CPUExecutionProvider").run(
                    None, {"W": raw, "I": np.array([0, 2, -1, 3, -2], dtype=numpy_type)}
                )[0]
                np.testing.assert_array_equal(got[:, 0], np.array([1, 3, 1, 1, 1], dtype=np.float32))

    def test_cpu_low_precision_chunked_rows(self):
        rng = np.random.default_rng(45)
        indices = np.array([4, 2, 0, 1], dtype=np.int64)
        for gtype, bel, bbytes in GGUF_TYPES_INFO:
            for k in (257, 768, 1025) if bel == 1 else (768,):
                raw = _make_synthetic_gguf_rows(rng, 5, k, gtype, bel, bbytes, quant_scale=2**-11)
                expected = _dequantize_weights(raw, k, gtype)[indices]
                for dtype in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
                    with self.subTest(gtype=gtype, k=k, dtype=dtype):
                        session = _session(_gather_model(5, k, raw.shape[1], gtype, dtype), "CPUExecutionProvider")
                        got = session.run(None, {"W": raw, "I": indices})[0]
                        np.testing.assert_array_equal(got, _round_to_dtype(expected, dtype))


class TestGatherPackedValidation(unittest.TestCase):
    def test_reject_row_byte_overflow(self):
        for gtype in (0, 1, 30):
            for k in ((1 << 62) + 1, (1 << 63) - 1):
                with self.subTest(gtype=gtype, k=k), self.assertRaisesRegex(Exception, "invalid K"):
                    _session(_gather_model(1, k, 4, gtype), "CPUExecutionProvider")

    def test_reject_invalid_output_dtype_before_narrowing(self):
        for dtype in (-1, 0, (1 << 32) + 1, (1 << 32) + 10, (1 << 32) + 16, (1 << 63) - 1):
            model = _gather_model(1, 1, 4, 0)
            next(attr for attr in model.graph.node[0].attribute if attr.name == "output_dtype").i = dtype
            with self.subTest(dtype=dtype), self.assertRaisesRegex(Exception, "output_dtype must be"):
                _session(model, "CPUExecutionProvider")

    def test_reject_partial_quantization_blocks(self):
        for provider in ("CPUExecutionProvider", "CUDAExecutionProvider"):
            if provider not in ort.get_available_providers():
                continue
            for gtype, k, row_bytes in ((2, 31, 18), (16, 255, 66), (42, 63, 18)):
                with self.subTest(provider=provider, gtype=gtype), self.assertRaisesRegex(Exception, "invalid K"):
                    _session(_gather_model(1, k, row_bytes, gtype), provider)

    @unittest.skipIf("CUDAExecutionProvider" not in ort.get_available_providers(), "CUDAExecutionProvider is required")
    def test_cuda_k_index_limit(self):
        max_k = 2147483391
        session = _session(_gather_model(0, max_k, max_k * 4, 0), "CUDAExecutionProvider")
        got = session.run(None, {"W": np.empty((0, max_k * 4), np.uint8), "I": np.empty(0, np.int64)})[0]
        self.assertEqual(got.shape, (0, max_k))
        for k in (max_k + 1, 1 << 31, (1 << 32) + 32):
            with self.subTest(k=k), self.assertRaisesRegex(Exception, "requires K <="):
                _session(_gather_model(0, k, k * 4, 0), "CUDAExecutionProvider")

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
    @classmethod
    def setUpClass(cls):
        cls.activation_dtypes = (
            (TensorProto.FLOAT16, TensorProto.BFLOAT16) if _cuda_bf16_supported() else (TensorProto.FLOAT16,)
        )

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
            for dtype in self.activation_dtypes:
                session = _session(_matmul_model(k, n, raw.shape[1], gtype, dtype), "CUDAExecutionProvider")
                for m in (1, 3, 7, 8, 9, 32):
                    with self.subTest(gtype=gtype, dtype=dtype, m=m):
                        a = _round_to_dtype(rng.standard_normal((m, k), dtype=np.float32) * 0.2, dtype)
                        got = session.run(None, {"A_f32": a, "B": raw})[0]
                        target = _round_to_dtype(weights, dtype)
                        ref = _round_to_dtype(a @ target.T, dtype)
                        np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)

    def test_matmul_packed_weight_rounding_across_dispatch(self):
        k = 32
        for dtype in self.activation_dtypes:
            for n in (1, 65535, 65536, 65537):
                weights = np.zeros((n, k), dtype=np.float32)
                weights[:, :2] = [1.00390625, -1.0]
                raw = weights.view(np.uint8).reshape(n, k * 4)
                for policy in (None, "0", "1"):
                    session = _session(_matmul_model(k, n, raw.shape[1], 0, dtype), "CUDAExecutionProvider", policy)
                    for m in (1, 8, 9):
                        with self.subTest(dtype=dtype, n=n, m=m, policy=policy):
                            got = session.run(None, {"A_f32": np.ones((m, k), dtype=np.float32), "B": raw})[0]
                            bf16_weights = dtype == TensorProto.BFLOAT16 and (policy != "1" or m > 8 or n >= 65536)
                            expected = 0.0 if bf16_weights else 0.00390625
                            np.testing.assert_array_equal(got, np.full((m, n), expected, dtype=np.float32))

    def test_matmul_packed_prefill_fp32_reduction(self):
        k, n = 4096, 32
        weights = np.ones((n, k), dtype=np.float32)
        weights[:, -1] = 1.125
        raw = weights.view(np.uint8).reshape(n, k * 4)
        for dtype in self.activation_dtypes:
            session = _session(_matmul_model(k, n, raw.shape[1], 0, dtype), "CUDAExecutionProvider")
            for m in (9, 32):
                with self.subTest(dtype=dtype, m=m):
                    a = np.ones((m, k), dtype=np.float32)
                    a[:, k // 2 :] = -1
                    got = session.run(None, {"A_f32": a, "B": raw})[0]
                    # Preserve the residual after cancellation of large partial sums.
                    np.testing.assert_array_equal(got, np.full((m, n), -0.125, dtype=np.float32))

    def test_matmul_packed_fp16_prefill_large_weights(self):
        q8_block = np.array([1024], dtype="<f2").tobytes() + bytes([127]) * 32
        q5_block = np.array([4096, 0], dtype="<f2").tobytes() + bytes([1]) * 12 + bytes([255]) * 32 + bytes([240]) * 128
        cases = [
            (0, 32, 4099, np.full(32, 100000, dtype="<f4").tobytes()),
            (30, 32, 3, _f32_to_bf16_u16(np.full(32, 100000, np.float32)).astype("<u2").tobytes()),
            (8, 32, 3, q8_block),
            (13, 256, 3, q5_block),
        ]
        for gtype, k, n, block in cases:
            raw = np.tile(np.frombuffer(block, np.uint8), (n, 1))
            weights = _dequantize_weights(raw, k, gtype)
            self.assertGreater(np.max(np.abs(weights)), np.finfo(np.float16).max)
            for policy in (None, "1"):
                session = _session(
                    _matmul_model(k, n, raw.shape[1], gtype, TensorProto.FLOAT16), "CUDAExecutionProvider", policy
                )
                for m in (1, 8, 9, 129):
                    with self.subTest(gtype=gtype, m=m, policy=policy):
                        a = _round_to_dtype(np.full((m, k), 1e-5, np.float32), TensorProto.FLOAT16)
                        got = session.run(None, {"A_f32": a, "B": raw})[0]
                        unrounded_gemv = policy == "1" and m <= 8
                        with np.errstate(over="ignore", invalid="ignore"):
                            target = weights if unrounded_gemv else _round_to_dtype(weights, TensorProto.FLOAT16)
                            ref = _round_to_dtype(a @ target.T, TensorProto.FLOAT16)
                        self.assertEqual(np.all(np.isfinite(got)), unrounded_gemv)
                        np.testing.assert_array_equal(got, ref)

    def test_matmul_packed_reject_partial_blocks_at_session_creation(self):
        for gtype, bel, bbytes in GGUF_TYPES_INFO:
            if bel <= 32:
                continue
            with self.subTest(gtype=gtype), self.assertRaisesRegex(Exception, "invalid K"):
                _session(_matmul_model(32, 1, bbytes, gtype, TensorProto.FLOAT16), "CUDAExecutionProvider")

    def test_matmul_packed_partial_tensor_core_tile(self):
        if TensorProto.BFLOAT16 not in self.activation_dtypes:
            self.skipTest("BF16 MatMulPacked requires compute capability 8.0 or higher")
        rng = np.random.default_rng(124)
        k, n = 32, 65537
        # F32 values must be rounded to BF16 before the Tensor Core multiplication.
        raw = _make_synthetic_gguf_rows(rng, n, k, 0, 1, 4)
        weights = _round_to_dtype(raw.view(np.float32).reshape(n, k), TensorProto.BFLOAT16)
        session = _session(_matmul_model(k, n, raw.shape[1], 0), "CUDAExecutionProvider")
        bound_model = _model(
            [helper.make_node("MatMulPacked", ["A", "B"], ["Y"], domain="com.microsoft", K=k, N=n, packed_type=0)],
            [
                helper.make_tensor_value_info("A", TensorProto.BFLOAT16, ["M", k]),
                helper.make_tensor_value_info("B", TensorProto.UINT8, [n, raw.shape[1]]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, ["M", n])],
        )
        bound_session = _session(bound_model, "CUDAExecutionProvider")
        for m in (1, 7, 8, 9, 32):
            with self.subTest(m=m):
                a = _round_to_dtype(rng.standard_normal((m, k), dtype=np.float32), TensorProto.BFLOAT16)
                got = session.run(None, {"A_f32": a, "B": raw})[0]
                ref = _round_to_dtype(a @ weights.T, TensorProto.BFLOAT16)
                np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)
                unaligned = _run_cuda_bound(
                    bound_session,
                    {
                        "A": (TensorProto.BFLOAT16, _activation_storage(a, TensorProto.BFLOAT16), 2),
                        "B": (TensorProto.UINT8, raw, 1),
                    },
                    (m, n),
                    TensorProto.BFLOAT16,
                )
                np.testing.assert_allclose(unaligned, ref, rtol=1e-2, atol=1e-2)

    def test_matmul_packed_tensor_core_multiblock_k_split(self):
        if TensorProto.BFLOAT16 not in self.activation_dtypes:
            self.skipTest("BF16 MatMulPacked requires compute capability 8.0 or higher")
        rng = np.random.default_rng(125)
        k, n = 512, 65537
        for gtype, bel, bbytes in GGUF_TYPES_INFO:
            if gtype not in (0, 8, 13, 14, 23):
                continue
            raw = _make_synthetic_gguf_rows(rng, n, k, gtype, bel, bbytes, quant_scale=2**-10)
            weights = _round_to_dtype(_dequantize_weights(raw, k, gtype), TensorProto.BFLOAT16)
            session = _session(_matmul_model(k, n, raw.shape[1], gtype), "CUDAExecutionProvider")
            for m in (1, 7, 8):
                with self.subTest(gtype=gtype, m=m):
                    a = _round_to_dtype(rng.standard_normal((m, k), dtype=np.float32) * 0.2, TensorProto.BFLOAT16)
                    got = session.run(None, {"A_f32": a, "B": raw})[0]
                    ref = _round_to_dtype(a @ weights.T, TensorProto.BFLOAT16)
                    np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)

    def test_matmul_packed_dimension_limits(self):
        max_dimension = 2147483616
        _session(_matmul_model(max_dimension, 1, max_dimension * 4, 0, TensorProto.FLOAT16), "CUDAExecutionProvider")
        _session(_matmul_model(32, max_dimension, 128, 0, TensorProto.FLOAT16), "CUDAExecutionProvider")
        for dimension in (max_dimension + 1, 1 << 31, (1 << 32) + 32):
            for k, n in ((dimension, 1), (32, dimension)):
                with self.subTest(k=k, n=n), self.assertRaisesRegex(Exception, "requires K and N <="):
                    _session(_matmul_model(k, n, k * 4, 0, TensorProto.FLOAT16), "CUDAExecutionProvider")
        with self.assertRaisesRegex(Exception, "K to be a multiple of 32"):
            _session(_matmul_model(31, 1, 124, 0, TensorProto.FLOAT16), "CUDAExecutionProvider")

    def test_moe_packed_dimension_limits(self):
        for dimension in (2147481600 + 32, 1 << 31, (1 << 32) + 32):
            for hidden_size, inter_size in ((dimension, 32), (32, dimension)):
                with self.subTest(hidden=hidden_size, inter=inter_size):
                    row_bytes = (hidden_size * 2, hidden_size * 2, inter_size * 2)
                    model = _moe_model(0, hidden_size, inter_size, 1, TensorProto.FLOAT16, (1, 1, 1), row_bytes, True)
                    session = _session(model, "CUDAExecutionProvider")
                    inputs = {
                        "X_f32": np.empty((0, hidden_size), np.float32),
                        "R_f32": np.empty((0, 0), np.float32),
                        "G": np.empty((0, inter_size, row_bytes[0]), np.uint8),
                        "U": np.empty((0, inter_size, row_bytes[1]), np.uint8),
                        "D": np.empty((0, hidden_size, row_bytes[2]), np.uint8),
                    }
                    with self.assertRaisesRegex(Exception, "hidden_size and inter_size in 1"):
                        session.run(None, inputs)

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

    def _check_moe(
        self, num_experts, hidden_size, inter_size, dtype, formats, row_counts, normalize, quant_scale=2**-7, top_k=4
    ):
        rng = np.random.default_rng(999)
        top_k = min(top_k, num_experts)
        raw_weights = []
        weights = []
        for out_size, k, (gtype, bel, bbytes) in zip(
            (inter_size, inter_size, hidden_size), (hidden_size, hidden_size, inter_size), formats, strict=True
        ):
            raw = _make_synthetic_gguf_rows(rng, num_experts * out_size, k, gtype, bel, bbytes, quant_scale)
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
                # High expert IDs check compact tile indexing and fill BF16 prefill subtiles.
                logits[:, -8:] += 4.0
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
        for dtype in self.activation_dtypes:
            for num_experts in (16, 512):
                for normalize in (False, True):
                    self._check_moe(num_experts, 256, 64, dtype, formats, (4, 48), normalize)

    def test_moe_packed_sparse_experts_at_prefill_boundary(self):
        for dtype in self.activation_dtypes:
            self._check_moe(512, 64, 64, dtype, ((1, 1, 2),) * 3, (16, 17, 48), True, top_k=8)

    def test_moe_packed_fp16_large_activations(self):
        formats = ((1, 1, 2),) * 3
        for hidden_size, inter_size in ((4096, 64), (256, 4096)):
            self._check_moe(2, hidden_size, inter_size, TensorProto.FLOAT16, formats, (4, 48), True)

    def test_moe_packed_bf16_shared_chunk_reuse(self):
        if TensorProto.BFLOAT16 not in self.activation_dtypes:
            self.skipTest("BF16 MoEPacked requires compute capability 8.0 or higher")
        # Bound intermediate-rounding noise without relaxing the shared tolerances.
        self._check_moe(4, 768, 768, TensorProto.BFLOAT16, ((16, 256, 66),) * 3, (4, 48), True, quant_scale=2**-10)

    def test_moe_packed_workspace_reuse_across_chunks(self):
        for dtype in self.activation_dtypes:
            self._check_moe(2, 32, 32, dtype, ((1, 1, 2),) * 3, (2049,), True)

    def test_moe_packed_lookup_and_mixed_formats(self):
        # Bound intermediate-rounding noise in near-zero outputs without relaxing tolerances.
        cases = [
            (((17, 256, 74), (17, 256, 74), (8, 32, 34)), 2**-10),
            (((22, 256, 82), (22, 256, 82), (20, 32, 18)), 2**-10),
            (((18, 256, 98), (18, 256, 98), (14, 256, 210)), 2**-11),
            (((16, 256, 66), (17, 256, 74), (13, 256, 176)), 2**-10),
        ]
        for dtype in self.activation_dtypes:
            for formats, quant_scale in cases:
                with self.subTest(dtype=dtype, formats=formats):
                    self._check_moe(2, 256, 256, dtype, formats, (4, 48), True, quant_scale=quant_scale)

    def test_gather_packed_unaligned_weights(self):
        rng = np.random.default_rng(44)
        for gtype, bel, bbytes in GGUF_TYPES_INFO:
            for k in (1, 8, 33, 256) if bel == 1 else (256,):
                with self.subTest(gtype=gtype, k=k):
                    raw = _make_synthetic_gguf_rows(rng, 3, k, gtype, bel, bbytes)
                    indices = np.array([2, 0, 1, -1, 3], dtype=np.int64)
                    model = _gather_model(3, k, raw.shape[1], gtype)
                    session = _session(model, "CUDAExecutionProvider")
                    # The float-to-float Cast is removed, exposing the bound GatherPacked output.
                    got = _run_cuda_bound(
                        session,
                        {"W": (TensorProto.UINT8, raw, 1), "I": (TensorProto.INT64, indices, 0)},
                        (5, k),
                        TensorProto.FLOAT,
                        output_offset=4,
                    )
                    ref = _session(model, "CPUExecutionProvider").run(None, {"W": raw, "I": indices})[0]
                    np.testing.assert_array_equal(got, ref)

    def test_matmul_packed_unaligned_buffers(self):
        rng = np.random.default_rng(125)
        k = 256
        for dtype in self.activation_dtypes:
            for n in (32, 1025):
                for gtype, bel, bbytes in GGUF_TYPES_INFO:
                    raw = _make_synthetic_gguf_rows(rng, n, k, gtype, bel, bbytes)
                    weights = _dequantize_weights(raw, k, gtype)
                    model = _model(
                        [
                            helper.make_node(
                                "MatMulPacked", ["A", "B"], ["Y"], domain="com.microsoft", K=k, N=n, packed_type=gtype
                            )
                        ],
                        [
                            helper.make_tensor_value_info("A", dtype, ["M", k]),
                            helper.make_tensor_value_info("B", TensorProto.UINT8, [n, raw.shape[1]]),
                        ],
                        [helper.make_tensor_value_info("Y", dtype, ["M", n])],
                    )
                    session = _session(model, "CUDAExecutionProvider")
                    for m in (1, 3, 8, 9):
                        with self.subTest(dtype=dtype, n=n, gtype=gtype, m=m):
                            a = rng.standard_normal((m, k), dtype=np.float32) * 0.2
                            got = _run_cuda_bound(
                                session,
                                {"A": (dtype, _activation_storage(a, dtype), 2), "B": (TensorProto.UINT8, raw, 1)},
                                (m, n),
                                dtype,
                            )
                            target = _round_to_dtype(weights, dtype)
                            ref = _round_to_dtype(_round_to_dtype(a, dtype) @ target.T, dtype)
                            np.testing.assert_allclose(got, ref, rtol=1e-2, atol=1e-2)

    def test_moe_packed_unaligned_buffers(self):
        rng = np.random.default_rng(1000)
        num_experts, hidden_size, inter_size = 2, 256, 256
        for dtype in self.activation_dtypes:
            for formats in (((1, 1, 2),) * 3, ((17, 256, 74), (17, 256, 74), (8, 32, 34))):
                raw_weights, weights = [], []
                for out_size, k, (gtype, bel, bbytes) in zip(
                    (inter_size, inter_size, hidden_size), (hidden_size, hidden_size, inter_size), formats, strict=True
                ):
                    raw = _make_synthetic_gguf_rows(rng, num_experts * out_size, k, gtype, bel, bbytes)
                    raw = raw.reshape(num_experts, out_size, -1)
                    raw_weights.append(raw)
                    weights.append(_dequantize_weights(raw, k, gtype))
                model = _model(
                    [
                        helper.make_node(
                            "MoEPacked",
                            ["X", "R", "G", "U", "D"],
                            ["Y"],
                            domain="com.microsoft",
                            k=2,
                            gate_type=formats[0][0],
                            up_type=formats[1][0],
                            down_type=formats[2][0],
                        )
                    ],
                    [
                        helper.make_tensor_value_info("X", dtype, ["M", hidden_size]),
                        helper.make_tensor_value_info("R", dtype, ["M", num_experts]),
                        *[
                            helper.make_tensor_value_info(name, TensorProto.UINT8, list(raw.shape))
                            for name, raw in zip(("G", "U", "D"), raw_weights, strict=True)
                        ],
                    ],
                    [helper.make_tensor_value_info("Y", dtype, ["M", hidden_size])],
                )
                session = _session(model, "CUDAExecutionProvider")
                for m in (4, 48):
                    with self.subTest(dtype=dtype, formats=formats, m=m):
                        x = rng.standard_normal((m, hidden_size), dtype=np.float32) * 0.2
                        logits = rng.standard_normal((m, num_experts), dtype=np.float32)
                        inputs = {
                            "X": (dtype, _activation_storage(x, dtype), 2),
                            "R": (dtype, _activation_storage(logits, dtype), 2),
                        }
                        inputs.update(
                            {
                                name: (TensorProto.UINT8, raw, 1)
                                for name, raw in zip(("G", "U", "D"), raw_weights, strict=True)
                            }
                        )
                        got = _run_cuda_bound(session, inputs, (m, hidden_size), dtype)
                        ref = _moe_reference(x, logits, *weights, 2, dtype, True)
                        np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-3)

    def _require_moe_offload(self):
        # MoE offload requires SM80+, which is exactly when BF16 activations are enabled for this suite.
        if TensorProto.BFLOAT16 not in self.activation_dtypes:
            self.skipTest("MoE offload requires compute capability 8.0 or higher")

    def test_moe_packed_cuda_offload_parity(self):
        self._require_moe_offload()
        num_experts = 16
        hidden_size = 256
        inter_size = 64
        top_k = 4
        formats = ((16, 256, 66), (16, 256, 66), (42, 64, 18))
        rng = np.random.default_rng(1234)
        raw_weights = []
        weights = []
        for out_size, k, (gtype, bel, bbytes) in zip(
            (inter_size, inter_size, hidden_size), (hidden_size, hidden_size, inter_size), formats, strict=True
        ):
            raw = _make_synthetic_gguf_rows(rng, num_experts * out_size, k, gtype, bel, bbytes)
            raw = raw.reshape(num_experts, out_size, -1)
            raw_weights.append(raw)
            weights.append(_dequantize_weights(raw, k, gtype))

        row_bytes = [w.shape[-1] for w in raw_weights]
        for dtype in self.activation_dtypes:
            model = _model(
                [
                    helper.make_node("Cast", ["X_f32"], ["X"], to=int(dtype)),
                    helper.make_node("Cast", ["R_f32"], ["R"], to=int(dtype)),
                    helper.make_node(
                        "MoEPacked",
                        ["X", "R", "G", "U", "D"],
                        ["Y"],
                        domain="com.microsoft",
                        k=top_k,
                        normalize_routing_weights=1,
                        activation_type="swiglu",
                        gate_type=formats[0][0],
                        up_type=formats[1][0],
                        down_type=formats[2][0],
                    ),
                    helper.make_node("Cast", ["Y"], ["Y_f32"], to=int(TensorProto.FLOAT)),
                ],
                [
                    helper.make_tensor_value_info("X_f32", TensorProto.FLOAT, ["M", hidden_size]),
                    helper.make_tensor_value_info("R_f32", TensorProto.FLOAT, ["M", num_experts]),
                ],
                [helper.make_tensor_value_info("Y_f32", TensorProto.FLOAT, ["M", hidden_size])],
            )
            for name, raw in zip(("G", "U", "D"), raw_weights, strict=True):
                model.graph.initializer.append(numpy_helper.from_array(raw, name=name))

            slot_bytes = sum(inter_size * rb for rb in row_bytes[:2]) + hidden_size * row_bytes[2]
            so = ort.SessionOptions()
            so.add_session_config_entry("session.moe.offload.mode", "required")
            so.add_session_config_entry("session.moe.offload.placement", "adaptive")
            so.add_session_config_entry("session.moe.offload.cpu_execution", "auto")
            so.add_session_config_entry("session.moe.offload.device_expert_bytes", str(6 * slot_bytes + 4096))
            sess_offload = ort.InferenceSession(
                model.SerializeToString(), sess_options=so, providers=["CUDAExecutionProvider"]
            )

            for m in (0, 1, 4, 32, 2049):
                x = rng.standard_normal((m, hidden_size), dtype=np.float32) * 0.2
                logits = rng.standard_normal((m, num_experts), dtype=np.float32)
                got = sess_offload.run(None, {"X_f32": x, "R_f32": logits})[0]
                ref = _moe_reference(x, logits, *weights, top_k, dtype, True)
                np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-3)

    @staticmethod
    def _offload_moe_weights(rng, num_experts, hidden_size, inter_size, formats, quant_scale=2**-7):
        raw_weights, weights = [], []
        for out_size, k, (gtype, bel, bbytes) in zip(
            (inter_size, inter_size, hidden_size), (hidden_size, hidden_size, inter_size), formats, strict=True
        ):
            raw = _make_synthetic_gguf_rows(rng, num_experts * out_size, k, gtype, bel, bbytes, quant_scale)
            raw = raw.reshape(num_experts, out_size, -1)
            raw_weights.append(raw)
            weights.append(_dequantize_weights(raw, k, gtype))
        slot_bytes = sum(w[0].nbytes for w in raw_weights)
        return raw_weights, weights, slot_bytes

    @staticmethod
    def _offload_moe_model(num_experts, hidden_size, top_k, dtype, formats, raw_weights):
        # X, R and Y are graph inputs/outputs so that I/O binding controls their device addresses.
        model = _model(
            [
                helper.make_node(
                    "MoEPacked",
                    ["X", "R", "G", "U", "D"],
                    ["Y"],
                    domain="com.microsoft",
                    k=top_k,
                    normalize_routing_weights=1,
                    activation_type="swiglu",
                    gate_type=formats[0][0],
                    up_type=formats[1][0],
                    down_type=formats[2][0],
                )
            ],
            [
                helper.make_tensor_value_info("X", dtype, ["M", hidden_size]),
                helper.make_tensor_value_info("R", dtype, ["M", num_experts]),
            ],
            [helper.make_tensor_value_info("Y", dtype, ["M", hidden_size])],
        )
        for name, raw in zip(("G", "U", "D"), raw_weights, strict=True):
            model.graph.initializer.append(numpy_helper.from_array(raw, name=name))
        return model

    @staticmethod
    def _offload_session(model, resident_bytes, provider_options=None):
        so = ort.SessionOptions()
        so.add_session_config_entry("session.moe.offload.mode", "required")
        so.add_session_config_entry("session.moe.offload.placement", "adaptive")
        so.add_session_config_entry("session.moe.offload.cpu_execution", "auto")
        so.add_session_config_entry("session.moe.offload.device_expert_bytes", str(resident_bytes))
        return ort.InferenceSession(
            model.SerializeToString(),
            sess_options=so,
            providers=[("CUDAExecutionProvider", provider_options or {})],
        )

    def test_moe_packed_cuda_offload_unaligned_buffers_and_small_prefill(self):
        self._require_moe_offload()
        # 64 experts with 8 resident: prefill with <= 128 routes checks the tile-descriptor workspace bound, and
        # element-aligned I/O-bound buffers check the cold-input copy and both offload finalizers.
        num_experts, hidden_size, inter_size, top_k = 64, 256, 64, 2
        formats = ((16, 256, 66), (16, 256, 66), (42, 64, 18))
        rng = np.random.default_rng(2024)
        raw_weights, weights, slot_bytes = self._offload_moe_weights(rng, num_experts, hidden_size, inter_size, formats)
        for dtype in self.activation_dtypes:
            model = self._offload_moe_model(num_experts, hidden_size, top_k, dtype, formats, raw_weights)
            session = self._offload_session(model, 8 * slot_bytes + 4096)
            for m in (1, 4, 17, 48):
                with self.subTest(dtype=dtype, m=m):
                    x = rng.standard_normal((m, hidden_size), dtype=np.float32) * 0.2
                    logits = rng.standard_normal((m, num_experts), dtype=np.float32)
                    inputs = {
                        "X": (dtype, _activation_storage(x, dtype), 2),
                        "R": (dtype, _activation_storage(logits, dtype), 2),
                    }
                    got = _run_cuda_bound(session, inputs, (m, hidden_size), dtype)
                    ref = _moe_reference(x, logits, *weights, top_k, dtype, True)
                    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-3)

    def test_moe_packed_cuda_offload_large_shapes(self):
        self._require_moe_offload()
        cases = [
            # FP16 hidden_size=4096 needs 64 KiB of FC1 activation staging, above the default shared-memory limit.
            (4, 4096, 64, 2, ((1, 1, 2), (1, 1, 2), (1, 1, 2)), 2**-7, (TensorProto.FLOAT16,)),
            # inter_size > 4096 with a down type that has no AVX2 kernel uses the scalar CPU dot product. A small
            # quantization scale bounds intermediate-rounding noise of the 4352-term sums.
            (2, 256, 4352, 1, ((16, 256, 66), (16, 256, 66), (8, 32, 34)), 2**-10, self.activation_dtypes),
        ]
        rng = np.random.default_rng(2025)
        for num_experts, hidden_size, inter_size, top_k, formats, quant_scale, dtypes in cases:
            raw_weights, weights, slot_bytes = self._offload_moe_weights(
                rng, num_experts, hidden_size, inter_size, formats, quant_scale
            )
            for dtype in dtypes:
                model = self._offload_moe_model(num_experts, hidden_size, top_k, dtype, formats, raw_weights)
                # No resident experts runs every decode route on the CPU; half resident mixes both paths.
                for resident in (0, num_experts // 2):
                    session = self._offload_session(model, resident * slot_bytes + 4096)
                    for m in (1, 4, 48):
                        with self.subTest(hidden=hidden_size, inter=inter_size, dtype=dtype, resident=resident, m=m):
                            x = rng.standard_normal((m, hidden_size), dtype=np.float32) * 0.2
                            logits = rng.standard_normal((m, num_experts), dtype=np.float32)
                            inputs = {
                                "X": (dtype, _activation_storage(x, dtype), 0),
                                "R": (dtype, _activation_storage(logits, dtype), 0),
                            }
                            got = _run_cuda_bound(session, inputs, (m, hidden_size), dtype, output_offset=0)
                            ref = _moe_reference(x, logits, *weights, top_k, dtype, True)
                            self.assertTrue(np.all(np.isfinite(got)))
                            self.assertGreater(np.max(np.abs(ref)), 1e-2)
                            np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-3)

    def test_moe_packed_cuda_offload_graph_replay_after_warmup(self):
        self._require_moe_offload()
        # A CUDA graph captured during warm-up runs must still compute cold experts when replayed by live runs.
        num_experts, hidden_size, inter_size, top_k, m = 16, 256, 64, 4, 2
        formats = ((16, 256, 66), (16, 256, 66), (42, 64, 18))
        rng = np.random.default_rng(2026)
        raw_weights, weights, slot_bytes = self._offload_moe_weights(rng, num_experts, hidden_size, inter_size, formats)
        dtype = TensorProto.FLOAT16
        model = self._offload_moe_model(num_experts, hidden_size, top_k, dtype, formats, raw_weights)
        session = self._offload_session(model, 4 * slot_bytes + 4096, {"enable_cuda_graph": "1"})

        x_value = ort.OrtValue.ortvalue_from_numpy(np.zeros((m, hidden_size), dtype=np.float16), "cuda", 0)
        r_value = ort.OrtValue.ortvalue_from_numpy(np.zeros((m, num_experts), dtype=np.float16), "cuda", 0)
        y_value = ort.OrtValue.ortvalue_from_numpy(np.zeros((m, hidden_size), dtype=np.float16), "cuda", 0)
        binding = session.io_binding()
        binding.bind_ortvalue_input("X", x_value)
        binding.bind_ortvalue_input("R", r_value)
        binding.bind_ortvalue_output("Y", y_value)

        warmup = ort.RunOptions()
        warmup.add_run_config_entry("ep.cuda.qmoe_in_warmup", "1")
        for _ in range(3):
            x_value.update_inplace(rng.standard_normal((m, hidden_size), dtype=np.float32).astype(np.float16) * 0.2)
            r_value.update_inplace(rng.standard_normal((m, num_experts), dtype=np.float32).astype(np.float16))
            session.run_with_iobinding(binding, warmup)

        for _ in range(3):
            x = rng.standard_normal((m, hidden_size), dtype=np.float32) * 0.2
            logits = rng.standard_normal((m, num_experts), dtype=np.float32)
            x_value.update_inplace(x.astype(np.float16))
            r_value.update_inplace(logits.astype(np.float16))
            session.run_with_iobinding(binding)
            got = y_value.numpy().astype(np.float32)
            ref = _moe_reference(x, logits, *weights, top_k, dtype, True)
            np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-3)

    # Formats that route decode work through every CPU cold-expert kernel family: dual-row and two-channel gate/up
    # kernels, four-column and single-row FC2 kernels, mixed gate/up types, and the scalar fallback.
    _CPU_COLD_EXPERT_FORMATS = (
        ((22, 256, 82), (22, 256, 82), (20, 32, 18)),  # IQ2_S / IQ4_NL
        ((16, 256, 66), (16, 256, 66), (42, 64, 18)),  # IQ2_XXS / Q2_0
        ((18, 256, 98), (18, 256, 98), (6, 32, 22)),  # IQ3_XXS / Q5_0
        ((17, 256, 74), (17, 256, 74), (23, 256, 136)),  # IQ2_XS / IQ4_XS
        ((23, 256, 136), (23, 256, 136), (22, 256, 82)),  # IQ4_XS / IQ2_S
        ((20, 32, 18), (42, 64, 18), (16, 256, 66)),  # IQ4_NL + Q2_0 / IQ2_XXS
        ((6, 32, 22), (18, 256, 98), (17, 256, 74)),  # Q5_0 + IQ3_XXS / IQ2_XS
        ((8, 32, 34), (2, 32, 18), (18, 256, 98)),  # Q8_0 + Q4_0 (scalar) / IQ3_XXS
    )

    def _assert_rounding_parity(self, got, expected, dtype):
        # With identical rounding points only rare one-ulp flips from summation order remain, while a missing or extra
        # rounding to the activation type changes a large fraction of the outputs.
        ulp = 2.0**-7 if dtype == TensorProto.BFLOAT16 else 2.0**-10
        row_scale = np.abs(expected).max(axis=-1, keepdims=True)
        self.assertGreater(float(row_scale.min()), 0.0)
        self.assertLessEqual(float(np.max(np.abs(got - expected) / (ulp * row_scale))), 2.0)
        self.assertLessEqual(float(np.mean(got != expected)), 0.05)

    def _run_offload_decode_rows(self, session, x, logits, dtype, offset=0):
        inputs = {
            "X": (dtype, _activation_storage(x, dtype), offset),
            "R": (dtype, _activation_storage(logits, dtype), offset),
        }
        return _run_cuda_bound(session, inputs, x.shape, dtype, output_offset=offset)

    def test_moe_packed_cuda_offload_cpu_kernels_match_reference(self):
        self._require_moe_offload()
        # No resident experts runs every decode route on the CPU, whose kernels keep the dequantized weights in FP32 and
        # round to the activation type where the CUDA kernels store it.
        num_experts, hidden_size, inter_size, top_k = 8, 256, 256, 4
        rng = np.random.default_rng(2027)
        for formats in self._CPU_COLD_EXPERT_FORMATS:
            raw_weights, weights, _ = self._offload_moe_weights(rng, num_experts, hidden_size, inter_size, formats)
            for dtype in self.activation_dtypes:
                model = self._offload_moe_model(num_experts, hidden_size, top_k, dtype, formats, raw_weights)
                session = self._offload_session(model, 4096)
                got, ref = [], []
                # One, three, and sixteen rows cover single-row, paired-row, and odd-row kernel tails.
                for m in (1, 3, 16):
                    x = rng.standard_normal((m, hidden_size), dtype=np.float32) * 0.2
                    logits = rng.standard_normal((m, num_experts), dtype=np.float32)
                    got.append(self._run_offload_decode_rows(session, x, logits, dtype))
                    # Large negative IQ4_XS gate sums overflow exp(-g); SwiGLU is then zero, as in the kernels.
                    with np.errstate(over="ignore"):
                        ref.append(_moe_reference(x, logits, *weights, top_k, dtype, True, round_weights=False))
                with self.subTest(formats=formats, dtype=dtype):
                    self._assert_rounding_parity(np.concatenate(got), np.concatenate(ref), dtype)

    def test_moe_packed_cuda_offload_mixed_placement_matches_gpu(self):
        self._require_moe_offload()
        # With half of the experts resident, most tokens combine GPU and CPU routes. FP16 CUDA and CPU kernels use the
        # same weights and rounding points, so the hybrid output must match the GPU-only output, which rounds the
        # combined routes once. Offset buffers select the element-wise input copy and finalize kernels.
        num_experts, hidden_size, inter_size, top_k = 8, 256, 256, 4
        dtype = TensorProto.FLOAT16
        rng = np.random.default_rng(2028)
        for formats in self._CPU_COLD_EXPERT_FORMATS[:2]:
            raw_weights, _, slot_bytes = self._offload_moe_weights(rng, num_experts, hidden_size, inter_size, formats)
            model = self._offload_moe_model(num_experts, hidden_size, top_k, dtype, formats, raw_weights)
            gpu_options = ort.SessionOptions()
            gpu_options.add_session_config_entry("session.moe.offload.mode", "disabled")
            gpu_session = ort.InferenceSession(
                model.SerializeToString(), sess_options=gpu_options, providers=["CUDAExecutionProvider"]
            )
            hybrid_session = self._offload_session(model, (num_experts // 2) * slot_bytes + 4096)
            for offset in (0, 2):
                got, expected = [], []
                for m in (1, 3, 16):
                    x = rng.standard_normal((m, hidden_size), dtype=np.float32) * 0.2
                    logits = rng.standard_normal((m, num_experts), dtype=np.float32)
                    got.append(self._run_offload_decode_rows(hybrid_session, x, logits, dtype, offset))
                    expected.append(self._run_offload_decode_rows(gpu_session, x, logits, dtype, offset))
                with self.subTest(formats=formats, offset=offset):
                    self._assert_rounding_parity(np.concatenate(got), np.concatenate(expected), dtype)


if __name__ == "__main__":
    unittest.main()
