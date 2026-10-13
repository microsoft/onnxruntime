# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Session-level coverage for the generic CUDA DynamicQuantizeMatMulNBits implementation."""

import unittest
from concurrent.futures import ThreadPoolExecutor

import ml_dtypes
import numpy as np
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort
from onnxruntime.capi import _pybind_state as ort_capi
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
from onnxruntime.quantization.quant_utils import QuantFormat


def make_model(scales=None, overridable=False, bits=None, m=2, k=17, n=3, block=16):
    weight_bits = 8 if bits is None else bits
    groups = (k + block - 1) // block
    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT16, [m, k]),
        helper.make_tensor_value_info("B", TensorProto.UINT8, [n, groups, block * weight_bits // 8]),
    ]
    initializers = []
    if scales is None or overridable:
        inputs.append(helper.make_tensor_value_info("scales", TensorProto.FLOAT, [n, groups]))
    if scales is not None:
        initializers.append(numpy_helper.from_array(scales, "scales"))
    node = helper.make_node(
        "DynamicQuantizeMatMulNBits",
        ["A", "B", "scales"],
        ["Y"],
        domain="com.microsoft",
        block_size=block,
        **({"bits": bits} if bits is not None else {}),
    )
    graph = helper.make_graph(
        [node],
        "dynamic_quantized_matmul",
        inputs,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [m, n])],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 1)], ir_version=8)


def make_session(model, capture=False):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(
        model.SerializeToString(),
        options,
        providers=[("CUDAExecutionProvider", {"enable_cuda_graph": str(int(capture))})],
    )


def constant_weight_model(m=33, k=144, n=136, block=512, runtime_scales=False, dynamic_shape=False):
    scales = None if runtime_scales else np.ones((n, 1), dtype=np.float32)
    model = make_model(scales, bits=8, m=m, k=k, n=n, block=block)
    model.graph.input.remove(model.graph.input[1])
    model.graph.initializer.append(numpy_helper.from_array(np.full((n, 1, block), 129, dtype=np.uint8), "B"))
    if dynamic_shape:
        for value in (model.graph.input[0], model.graph.output[0]):
            value.type.tensor_type.shape.dim[0].ClearField("dim_value")
            value.type.tensor_type.shape.dim[0].dim_param = "M"
        model.graph.input[0].type.tensor_type.shape.dim[1].ClearField("dim_value")
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_param = "K"
    return model


def reference(a, packed, scales, bits, block, mode, activation_block, output_dtype=np.float16):
    a = a.astype(np.float32)
    m, k = a.shape
    n = packed.shape[0]
    pack = 8 // bits
    weights = np.stack([(packed >> (bits * lane)) & ((1 << bits) - 1) for lane in range(pack)], axis=-1)
    weights = weights.reshape(n, -1)[:, :k].astype(np.int32) - (1 << (bits - 1))
    domain_size = activation_block if mode == "dynamic_per_block" else k
    quantized = np.empty((m, k), dtype=np.int32)
    a_scales = np.empty((m, k), dtype=np.float32)
    for start in range(0, k, domain_size):
        end = min(k, start + domain_size)
        values = a[:, start:end]
        maximum = np.max(np.abs(a)) if mode == "dynamic_per_tensor" else np.max(np.abs(values), axis=1, keepdims=True)
        scale = np.where(maximum == 0, np.float32(1), maximum / np.float32(127)).astype(np.float32)
        quantized[:, start:end] = np.clip(np.rint(values / scale), -127, 127).astype(np.int32)
        a_scales[:, start:end] = scale
    output = None
    start = 0
    while start < k:
        end = min(k, start + block - start % block, start + domain_size - start % domain_size)
        # Exact in float64 (|acc| < 2**53), so it rounds to FP32 identically to an INT32 accumulator.
        acc = quantized[:, start:end].astype(np.float64) @ weights[:, start:end].T.astype(np.float64)
        term = (acc.astype(np.float32) * a_scales[:, start, None]) * scales[:, start // block].astype(np.float32)
        output = term if output is None else output + term
        start = end
    return (output + np.float32(0)).astype(output_dtype)


def prepack_sm80(packed, bits):
    """Applies the SM80 mixed-GEMM stages: row permutation, column-interleaved tiles, in-word pair interleave."""
    n, groups, blob = packed.shape
    k = groups * blob * 8 // bits
    codes = ((packed[..., None] >> (bits * np.arange(8 // bits))) & ((1 << bits) - 1)).reshape(n, k)
    interleave = 16 // bits
    permute_rows = 8 * interleave
    elements_per_word = 32 // bits
    words_per_tile = 64 // elements_per_word
    permutation = [2 * (r // (2 * interleave)) + 8 * ((r % (2 * interleave)) // 2) + r % 2 for r in range(permute_rows)]
    source_rows = (np.arange(k) // permute_rows) * permute_rows + np.tile(permutation, k // permute_rows)
    elements = codes[:, source_rows].reshape(n, -1, elements_per_word).astype(np.uint32)
    words = np.zeros(elements.shape[:2], dtype=np.uint32)
    for element in range(elements_per_word):
        slot = element // 2 if element % 2 == 0 else (element - 1) // 2 + elements_per_word // 2
        words |= elements[:, :, element] << np.uint32(slot * bits)
    column = np.arange(n)[:, None]
    word = np.arange(words.shape[1])[None, :]
    destination = (
        interleave * (word // words_per_tile * words_per_tile)
        + words_per_tile * (column % interleave)
        + word % words_per_tile
    )
    output = np.zeros((n // interleave, words.shape[1] * interleave), dtype=np.uint32)
    output[np.broadcast_to(column // interleave, words.shape), destination] = words
    return output.reshape(-1).view(np.uint8).reshape(packed.shape)


class TestDynamicQuantizeMatMulNBits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise unittest.SkipTest("CUDAExecutionProvider is required")

    def test_runtime_scales_are_revalidated(self):
        session = make_session(make_model())
        feeds = {
            "A": np.ones((2, 17), dtype=np.float16),
            "B": np.full((3, 2, 16), 129, dtype=np.uint8),
            "scales": np.ones((3, 2), dtype=np.float32),
        }
        np.testing.assert_array_equal(session.run(None, feeds)[0], np.full((2, 3), 17, dtype=np.float16))
        feeds["scales"][0, 0] = np.nan
        with self.assertRaisesRegex(ort_capi.InvalidArgument, "scales must contain only finite values"):
            session.run(None, feeds)
        feeds["scales"][:] = -1
        np.testing.assert_array_equal(session.run(None, feeds)[0], np.full((2, 3), -17, dtype=np.float16))

    def test_hopper_constant_weights_capture_and_rebinding(self):
        m, k, n = 129, 272, 512
        model = constant_weight_model(m, k, n)
        for capture in (False, True):
            with self.subTest(capture=capture):
                session = make_session(model, capture=capture)
                a = ort.OrtValue.ortvalue_from_numpy(np.ones((m, k), dtype=np.float16), "cuda", 0)
                y = ort.OrtValue.ortvalue_from_shape_and_type((m, n), np.float16, "cuda", 0)
                binding = session.io_binding()
                binding.bind_ortvalue_input("A", a)
                for value in range(1, 5):
                    if not capture:
                        y = ort.OrtValue.ortvalue_from_shape_and_type((m, n), np.float16, "cuda", 0)
                    a.update_inplace(np.full((m, k), value, dtype=np.float16))
                    binding.bind_ortvalue_output("Y", y)
                    session.run_with_iobinding(binding)
                    binding.synchronize_outputs()
                    np.testing.assert_array_equal(y.numpy(), np.full((m, n), value * k, dtype=np.float16))

    def test_hopper_random_weights_and_scale_types_match_exact_reference(self):
        rng = np.random.default_rng(42)
        for m, k, n in ((33, 144, 136), (129, 272, 512)):
            a = rng.normal(size=(m, k)).astype(np.float16)
            a[0] = 0
            packed = rng.integers(0, 256, size=(n, 1, 512), dtype=np.uint8)
            for dtype in (np.float16, np.float32, ml_dtypes.bfloat16):
                scales = rng.uniform(-0.02, 0.02, size=(n, 1)).astype(dtype)
                for mode in ("dynamic_per_tensor", "dynamic_per_token"):
                    with self.subTest(m=m, dtype=dtype, mode=mode):
                        model = make_model(scales, bits=8, m=m, k=k, n=n, block=512)
                        model.graph.input.remove(model.graph.input[1])
                        model.graph.initializer.append(numpy_helper.from_array(packed, "B"))
                        model.graph.node[0].attribute.append(helper.make_attribute("activation_quantization", mode))
                        expected = reference(a, packed, scales, 8, 512, mode, k)
                        result = make_session(model).run(None, {"A": a})[0]
                        np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))

    def test_sm80_prepacked_matches_exact_reference(self):
        if ort.get_device() != "GPU":
            self.skipTest("CUDAExecutionProvider is required")
        rng = np.random.default_rng(43)
        for bits, block in ((2, 64), (4, 32), (8, 32)):
            k, n = 256, 128
            packed = rng.integers(0, 256, size=(n, k // block, block * bits // 8), dtype=np.uint8)
            prepacked = prepack_sm80(packed, bits)
            for m, scale_dtype, mode in (
                (8, np.float32, "dynamic_per_token"),
                (9, np.float16, "dynamic_per_token"),
                (33, ml_dtypes.bfloat16, "dynamic_per_tensor"),
            ):
                with self.subTest(bits=bits, m=m, scale_dtype=scale_dtype, mode=mode):
                    a = rng.normal(size=(m, k)).astype(np.float16)
                    scales = rng.uniform(-0.02, 0.02, size=(n, k // block)).astype(scale_dtype)
                    model = make_model(scales, bits=bits, m=m, k=k, n=n, block=block)
                    model.graph.input.remove(model.graph.input[1])
                    model.graph.initializer.append(numpy_helper.from_array(prepacked, "B"))
                    model.graph.node[0].attribute.extend(
                        [
                            helper.make_attribute("activation_quantization", mode),
                            helper.make_attribute("weight_prepacked", 1),
                        ]
                    )
                    expected = reference(a, packed, scales, bits, block, mode, k)
                    try:
                        session = make_session(model)
                    except ort_capi.Fail as error:
                        if "SM80+" in str(error):
                            self.skipTest("SM80+ is required for prepacked DynamicQuantizeMatMulNBits")
                        raise
                    result = session.run(None, {"A": a})[0]
                    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))

    def test_sm80_prepacked_row_fragment_tiles_match_exact_reference(self):
        if ort.get_device() != "GPU":
            self.skipTest("CUDAExecutionProvider is required")
        rng = np.random.default_rng(44)
        k, n = 256, 1024
        for bits, block in ((2, 64), (4, 32), (8, 32)):
            packed = rng.integers(0, 256, size=(n, k // block, block * bits // 8), dtype=np.uint8)
            prepacked = prepack_sm80(packed, bits)
            scales = rng.uniform(-0.02, 0.02, size=(n, k // block)).astype(np.float16)
            for m in (17, 63, 64, 65, 127, 128, 129, 200, 333, 400, 513):
                with self.subTest(bits=bits, m=m):
                    a = rng.normal(size=(m, k)).astype(np.float16)
                    model = make_model(scales, bits=bits, m=m, k=k, n=n, block=block)
                    model.graph.input.remove(model.graph.input[1])
                    model.graph.initializer.append(numpy_helper.from_array(prepacked, "B"))
                    model.graph.node[0].attribute.extend(
                        [
                            helper.make_attribute("activation_quantization", "dynamic_per_token"),
                            helper.make_attribute("weight_prepacked", 1),
                        ]
                    )
                    expected = reference(a, packed, scales, bits, block, "dynamic_per_token", k)
                    try:
                        session = make_session(model)
                    except ort_capi.Fail as error:
                        if "SM80+" in str(error):
                            self.skipTest("SM80+ is required for prepacked DynamicQuantizeMatMulNBits")
                        raise
                    result = session.run(None, {"A": a})[0]
                    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))

    def test_standard_and_prepacked_row_tiles_capture_with_bias_match_exact_reference(self):
        rng = np.random.default_rng(46)
        k, n = 256, 4096
        bias = rng.uniform(-0.02, 0.02, size=n).astype(np.float16)
        for bits, block in ((2, 64), (4, 32), (8, 32)):
            packed = rng.integers(0, 256, size=(n, k // block, block * bits // 8), dtype=np.uint8)
            prepacked = prepack_sm80(packed, bits)
            scales = rng.uniform(-0.02, 0.02, size=(n, k // block)).astype(np.float16)
            for m in (1, 8, 9, 16, 32, 64, 65, 128, 129):
                for mode in ("dynamic_per_token", "dynamic_per_tensor"):
                    for weight_prepacked, weights in ((0, packed), (1, prepacked)):
                        with self.subTest(bits=bits, m=m, mode=mode, prepacked=weight_prepacked):
                            model = make_model(scales, bits=bits, m=m, k=k, n=n, block=block)
                            model.graph.input.remove(model.graph.input[1])
                            model.graph.initializer.extend(
                                [numpy_helper.from_array(weights, "B"), numpy_helper.from_array(bias, "bias")]
                            )
                            model.graph.node[0].input.append("bias")
                            model.graph.node[0].attribute.extend(
                                [
                                    helper.make_attribute("activation_quantization", mode),
                                    helper.make_attribute("weight_prepacked", weight_prepacked),
                                ]
                            )
                            try:
                                session = make_session(model, capture=True)
                            except ort_capi.Fail as error:
                                if "SM80+" in str(error):
                                    self.skipTest("SM80+ is required for prepacked DynamicQuantizeMatMulNBits")
                                raise
                            a = ort.OrtValue.ortvalue_from_shape_and_type((m, k), np.float16, "cuda", 0)
                            y = ort.OrtValue.ortvalue_from_shape_and_type((m, n), np.float16, "cuda", 0)
                            binding = session.io_binding()
                            binding.bind_ortvalue_input("A", a)
                            binding.bind_ortvalue_output("Y", y)
                            for iteration in range(4):
                                values = (rng.normal(size=(m, k)) * 4.0 ** (iteration - 1)).astype(np.float16)
                                values[iteration % m] = 0
                                a.update_inplace(values)
                                session.run_with_iobinding(binding)
                                binding.synchronize_outputs()
                                expected = reference(
                                    values, packed, scales, bits, block, mode, k, output_dtype=np.float32
                                )
                                expected = (expected + bias.astype(np.float32)).astype(np.float16)
                                np.testing.assert_array_equal(y.numpy().view(np.uint16), expected.view(np.uint16))

    def test_bfloat16_standard_and_prepacked_tiles_match_exact_reference(self):
        rng = np.random.default_rng(45)
        m, k, n = 513, 128, 1024
        a = rng.normal(size=(m, k)).astype(ml_dtypes.bfloat16)
        a[0] = 0
        a_value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(a.view(np.uint16), TensorProto.BFLOAT16)
        for bits, block in ((2, 64), (4, 32), (8, 32)):
            packed = rng.integers(0, 256, size=(n, k // block, block * bits // 8), dtype=np.uint8)
            prepacked = prepack_sm80(packed, bits)
            for scale_dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
                scales = rng.uniform(-0.02, 0.02, size=(n, k // block)).astype(scale_dtype)
                for mode in ("dynamic_per_token", "dynamic_per_tensor"):
                    expected = reference(a, packed, scales, bits, block, mode, k, ml_dtypes.bfloat16)
                    for weight_prepacked, weights in ((0, packed), (1, prepacked)):
                        with self.subTest(bits=bits, scales=scale_dtype, mode=mode, prepacked=weight_prepacked):
                            model = make_model(scales, bits=bits, m=m, k=k, n=n, block=block)
                            model.graph.input[0].type.tensor_type.elem_type = TensorProto.BFLOAT16
                            model.graph.output[0].type.tensor_type.elem_type = TensorProto.BFLOAT16
                            model.graph.input.remove(model.graph.input[1])
                            model.graph.initializer.append(numpy_helper.from_array(weights, "B"))
                            model.graph.node[0].attribute.extend(
                                [
                                    helper.make_attribute("activation_quantization", mode),
                                    helper.make_attribute("weight_prepacked", weight_prepacked),
                                ]
                            )
                            try:
                                session = make_session(model)
                            except ort_capi.Fail as error:
                                if "SM80+" in str(error):
                                    self.skipTest("SM80+ is required for prepacked DynamicQuantizeMatMulNBits")
                                raise
                            result = np.empty((m, n), dtype=np.uint16)
                            y_value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(result, TensorProto.BFLOAT16)
                            binding = session.io_binding()
                            binding.bind_ortvalue_input("A", a_value)
                            binding.bind_ortvalue_output("Y", y_value)
                            session.run_with_iobinding(binding)
                            binding.synchronize_outputs()
                            np.testing.assert_array_equal(result, expected.view(np.uint16))

    def test_hopper_constant_weights_concurrent_dynamic_shapes(self):
        n = 512
        session = make_session(constant_weight_model(n=n, dynamic_shape=True))
        shapes = ((17, 144), (65, 272), (129, 256), (15, 257))

        def run(index):
            m, k = shapes[index % len(shapes)]
            value = index + 1
            result = session.run(None, {"A": np.full((m, k), value, dtype=np.float16)})[0]
            np.testing.assert_array_equal(result, np.full((m, n), value * k, dtype=np.float16))

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(run, range(12)))

    def test_hopper_constant_weights_revalidate_runtime_scales(self):
        m, k, n = 129, 272, 512
        session = make_session(constant_weight_model(m, k, n, runtime_scales=True))
        feeds = {
            "A": np.ones((m, k), dtype=np.float16),
            "scales": np.ones((n, 1), dtype=np.float32),
        }
        np.testing.assert_array_equal(session.run(None, feeds)[0], np.full((m, n), k, dtype=np.float16))
        feeds["scales"][0, 0] = np.nan
        with self.assertRaisesRegex(ort_capi.InvalidArgument, "scales must contain only finite values"):
            session.run(None, feeds)
        feeds["scales"][:] = -1
        np.testing.assert_array_equal(session.run(None, feeds)[0], np.full((m, n), -k, dtype=np.float16))

    def test_grouped_packed_weights_match_exact_reference(self):
        rng = np.random.default_rng(0)
        block = 128
        activation_block = 192
        shapes = (
            (8, 256, 64),
            (16, 128, 512),
            (65, 320, 137),
            (961, 192, 961),
            (128, 256, 128),
            (1024, 256, 1024),
        )
        for m, k, n in shapes:
            groups = (k + block - 1) // block
            a = rng.normal(size=(m, k)).astype(np.float16)
            a[0] = 0
            scales = rng.uniform(-0.01, 0.01, size=(n, groups)).astype(np.float32)
            for bits in (2, 4, 8):
                packed = rng.integers(0, 256, size=(n, groups, block * bits // 8), dtype=np.uint8)
                for mode in ("dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"):
                    with self.subTest(bits=bits, mode=mode, m=m):
                        model = make_model(scales, bits=bits, m=m, k=k, n=n, block=block)
                        node = model.graph.node[0]
                        node.attribute.append(helper.make_attribute("activation_quantization", mode))
                        if mode == "dynamic_per_block":
                            node.attribute.append(helper.make_attribute("activation_block_size", activation_block))
                        result = make_session(model).run(None, {"A": a, "B": packed})[0]
                        expected = reference(a, packed, scales, bits, block, mode, activation_block)
                        np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))

    def test_row_counts_across_grouped_tile_selection_match_exact_reference(self):
        rng = np.random.default_rng(7)
        geometries = (
            (256, 96, 128, "dynamic_per_token", 0),
            (384, 72, 128, "dynamic_per_tensor", 0),
            (512, 64, 256, "dynamic_per_block", 256),
            (320, 64, 64, "dynamic_per_token", 0),
            (256, 128, 256, "dynamic_per_token", 0),
            (256, 96, 32, "dynamic_per_token", 0),
            (384, 128, 32, "dynamic_per_block", 96),
            (384, 64, 128, "dynamic_per_block", 96),
        )
        row_counts = (1, 8, 9, 16, 31, 32, 33, 49, 63, 64, 65, 97, 129, 193, 257, 641)
        for k, n, block, mode, activation_block in geometries:
            groups = (k + block - 1) // block
            scales = rng.uniform(-0.01, 0.01, size=(n, groups)).astype(np.float32)
            for bits in (2, 4, 8):
                packed = rng.integers(0, 256, size=(n, groups, block * bits // 8), dtype=np.uint8)
                for m in row_counts:
                    with self.subTest(bits=bits, k=k, n=n, block=block, mode=mode, m=m):
                        a = rng.normal(size=(m, k)).astype(np.float16)
                        a[0] = 0
                        model = make_model(scales, bits=bits, m=m, k=k, n=n, block=block)
                        node = model.graph.node[0]
                        node.attribute.append(helper.make_attribute("activation_quantization", mode))
                        if mode == "dynamic_per_block":
                            node.attribute.append(helper.make_attribute("activation_block_size", activation_block))
                        result = make_session(model).run(None, {"A": a, "B": packed})[0]
                        expected = reference(a, packed, scales, bits, block, mode, activation_block)
                        np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))

    def test_capture_replays_changed_activations_and_weights(self):
        for m, k, n, block in (
            (2, 17, 3, 16),
            (33, 64, 37, 128),
            (129, 272, 137, 512),
            (65, 320, 137, 128),
            (128, 256, 128, 128),
        ):
            with self.subTest(m=m, k=k):
                groups = (k + block - 1) // block
                session = make_session(
                    make_model(np.ones((n, groups), dtype=np.float32), m=m, k=k, n=n, block=block), capture=True
                )
                a = ort.OrtValue.ortvalue_from_numpy(np.ones((m, k), dtype=np.float16), "cuda", 0)
                b = ort.OrtValue.ortvalue_from_numpy(np.full((n, groups, block), 129, dtype=np.uint8), "cuda", 0)
                y = ort.OrtValue.ortvalue_from_shape_and_type((m, n), np.float16, "cuda", 0)
                binding = session.io_binding()
                binding.bind_ortvalue_input("A", a)
                binding.bind_ortvalue_input("B", b)
                binding.bind_ortvalue_output("Y", y)
                for iteration in range(5):
                    a.update_inplace(np.full((m, k), iteration + 1, dtype=np.float16))
                    b.update_inplace(np.full((n, groups, block), 129 + iteration, dtype=np.uint8))
                    session.run_with_iobinding(binding)
                    binding.synchronize_outputs()
                    expected = np.full((m, n), k * (iteration + 1) ** 2, dtype=np.float16)
                    np.testing.assert_array_equal(y.numpy(), expected)

    def test_capture_replays_match_exact_reference_for_native_paths(self):
        rng = np.random.default_rng(7)
        # (bits, mode, m, k, n, block): Hopper W8 (both tiles), tensor-core W8/W4/W2, and per-block intervals.
        cases = (
            (8, "dynamic_per_token", 8, 256, 128, 32),
            (4, "dynamic_per_token", 9, 256, 128, 32),
            (2, "dynamic_per_tensor", 1, 256, 128, 64),
            (4, "dynamic_per_block", 65, 384, 128, 32),
            (8, "dynamic_per_token", 256, 1024, 1024, 1024),
            (8, "dynamic_per_tensor", 256, 1024, 1024, 1024),
            (8, "dynamic_per_token", 129, 272, 512, 512),
            (8, "dynamic_per_token", 160, 512, 4096, 512),
            (4, "dynamic_per_token", 64, 512, 256, 128),
            (2, "dynamic_per_tensor", 64, 512, 256, 128),
            (4, "dynamic_per_block", 65, 320, 137, 128),
        )
        for bits, mode, m, k, n, block in cases:
            for constant_weights in (True, False):
                with self.subTest(bits=bits, mode=mode, m=m, n=n, constant_weights=constant_weights):
                    groups = (k + block - 1) // block
                    scales = rng.uniform(-0.01, 0.01, size=(n, groups)).astype(np.float32)
                    packed = rng.integers(0, 256, size=(n, groups, block * bits // 8), dtype=np.uint8)
                    model = make_model(scales, bits=bits, m=m, k=k, n=n, block=block)
                    node = model.graph.node[0]
                    node.attribute.append(helper.make_attribute("activation_quantization", mode))
                    if mode == "dynamic_per_block":
                        node.attribute.append(helper.make_attribute("activation_block_size", 192))
                    if constant_weights:
                        model.graph.input.remove(model.graph.input[1])
                        model.graph.initializer.append(numpy_helper.from_array(packed, "B"))
                    session = make_session(model, capture=True)
                    a = ort.OrtValue.ortvalue_from_numpy(np.zeros((m, k), dtype=np.float16), "cuda", 0)
                    y = ort.OrtValue.ortvalue_from_shape_and_type((m, n), np.float16, "cuda", 0)
                    binding = session.io_binding()
                    binding.bind_ortvalue_input("A", a)
                    if not constant_weights:
                        b = ort.OrtValue.ortvalue_from_numpy(packed, "cuda", 0)
                        binding.bind_ortvalue_input("B", b)
                    binding.bind_ortvalue_output("Y", y)
                    # Runs beyond the first replay one captured graph with new activations (including a zero row).
                    for iteration in range(4):
                        values = (rng.normal(size=(m, k)) * 4.0 ** (iteration - 1)).astype(np.float16)
                        values[iteration % m] = 0
                        a.update_inplace(values)
                        session.run_with_iobinding(binding)
                        binding.synchronize_outputs()
                        expected = reference(values, packed, scales, bits, block, mode, 192)
                        np.testing.assert_array_equal(y.numpy().view(np.uint16), expected.view(np.uint16))

    def test_concurrent_runs_use_independent_activation_scratch(self):
        for m, k, n, block in (
            (2, 17, 3, 16),
            (33, 64, 37, 128),
            (129, 272, 137, 512),
            (65, 320, 137, 128),
            (128, 256, 128, 128),
        ):
            with self.subTest(m=m, k=k):
                groups = (k + block - 1) // block
                session = make_session(make_model(np.ones((n, groups), dtype=np.float32), m=m, k=k, n=n, block=block))

                def run(value, session=session, m=m, k=k, n=n, groups=groups, block=block):
                    feeds = {
                        "A": np.full((m, k), value, dtype=np.float16),
                        "B": np.full((n, groups, block), 129, dtype=np.uint8),
                    }
                    return session.run(None, feeds)[0]

                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(run, range(1, 17)))
                for value, result in enumerate(results, 1):
                    np.testing.assert_array_equal(result, np.full((m, n), value * k, dtype=np.float16))

    def test_capture_rejects_mutable_scales_after_warmup(self):
        for overridable in (False, True):
            with self.subTest(overridable=overridable):
                scales = np.ones((3, 2), dtype=np.float32)
                model = make_model(scales if overridable else None, overridable)
                session = make_session(model, capture=True)
                values = {
                    "A": ort.OrtValue.ortvalue_from_numpy(np.ones((2, 17), dtype=np.float16), "cuda", 0),
                    "B": ort.OrtValue.ortvalue_from_numpy(np.full((3, 2, 16), 129, dtype=np.uint8), "cuda", 0),
                    "scales": ort.OrtValue.ortvalue_from_numpy(scales, "cuda", 0),
                }
                y = ort.OrtValue.ortvalue_from_shape_and_type((2, 3), np.float16, "cuda", 0)
                binding = session.io_binding()
                for name, value in values.items():
                    binding.bind_ortvalue_input(name, value)
                binding.bind_ortvalue_output("Y", y)
                warmup = ort.RunOptions()
                warmup.add_run_config_entry("gpu_graph_id", "-1")
                session.run_with_iobinding(binding, warmup)
                # ORT may perform warmup runs before attempting capture.
                with self.assertRaisesRegex(Exception, "CUDA graph capture requires non-overridable constant scales"):
                    for _ in range(4):
                        session.run_with_iobinding(binding)

    def test_offline_quantizer_initializer_compatibility(self):
        for bits in (2, 4, 8):
            with self.subTest(bits=bits):
                source = np.ones((32, 3), dtype=np.float32)
                packed = np.zeros((3, 2, 16 * bits // 8), dtype=np.uint8)
                scales = np.zeros((3, 2), dtype=np.float32)
                zero_points = np.zeros((3, 2), dtype=np.uint8)
                quantize = getattr(ort_capi, f"quantize_matmul_{bits}bits")
                quantize(packed, source, scales, zero_points, 16, 3, 17, True)
                model = make_model(scales, bits=bits)
                session = make_session(model)
                result = session.run(None, {"A": np.ones((2, 17), dtype=np.float16), "B": packed})[0]
                np.testing.assert_array_equal(result, np.full((2, 3), 17, dtype=np.float16))

    def test_export_opt_in_preserves_initializers_and_runs_native_geometry(self):
        rng = np.random.default_rng(17)
        a = rng.normal(size=(33, 65)).astype(np.float16)
        weights = rng.normal(size=(65, 37)).astype(np.float16)
        source = helper.make_model(
            helper.make_graph(
                [helper.make_node("MatMul", ["A", "weight"], ["Y"], name="projection")],
                "export",
                [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [33, 65])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [33, 37])],
                [numpy_helper.from_array(weights, "weight")],
            ),
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        for bits in (2, 4, 8):
            weight_only = MatMulNBitsQuantizer(
                type(source).FromString(source.SerializeToString()),
                algo_config=DefaultWeightOnlyQuantConfig(bits=bits, block_size=128, is_symmetric=True),
            )
            weight_only.process()
            for mode in ("dynamic_per_tensor", "dynamic_per_token", "dynamic_per_block"):
                with self.subTest(bits=bits, mode=mode):
                    quantizer = MatMulNBitsQuantizer(
                        type(source).FromString(source.SerializeToString()),
                        nodes_to_include=["projection"],
                        algo_config=DefaultWeightOnlyQuantConfig(
                            bits=bits,
                            block_size=128,
                            is_symmetric=True,
                            activation_quantization=mode,
                            activation_block_size=32 if mode == "dynamic_per_block" else None,
                        ),
                    )
                    quantizer.process()
                    model = quantizer.model.model
                    self.assertEqual(model.graph.node[0].op_type, "DynamicQuantizeMatMulNBits")
                    self.assertEqual(
                        [tensor.SerializeToString() for tensor in model.graph.initializer],
                        [tensor.SerializeToString() for tensor in weight_only.model.model.graph.initializer],
                    )
                    attributes = {attr.name: helper.get_attribute_value(attr) for attr in model.graph.node[0].attribute}
                    self.assertNotIn("K", attributes)
                    self.assertNotIn("N", attributes)
                    self.assertEqual(attributes["bits"], bits)
                    session = make_session(model)
                    result = session.run(None, {"A": a})[0]
                    packed, scales = [numpy_helper.to_array(tensor) for tensor in model.graph.initializer]
                    expected = reference(a, packed, scales, bits, 128, mode, 32)
                    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))


class TestDynamicQuantizeMatMulNBitsExportValidation(unittest.TestCase):
    def test_rejects_incompatible_export_settings(self):
        invalid = [
            {"is_symmetric": False},
            {"quant_format": QuantFormat.QDQ},
            {"activation_quantization": "none"},
            {"activation_block_size": 32},
            {"activation_quantization": "dynamic_per_block"},
            {"activation_quantization": "dynamic_per_block", "activation_block_size": 0},
            {"block_size": 15},
            {"bits": 2, "block_size": 512},
            {"accuracy_level": 1},
            {"op_types_to_quantize": ("MatMul", "Gather")},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                kwargs = {"is_symmetric": True, "activation_quantization": "dynamic_per_token", **overrides}
                with self.assertRaises(ValueError):
                    DefaultWeightOnlyQuantConfig(**kwargs)

    def test_rejects_fp32_activation_without_changing_dtype(self):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("MatMul", ["A", "B"], ["Y"], name="fp32")],
                "unsupported_activation",
                [helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 17])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
                [numpy_helper.from_array(np.ones((17, 3), dtype=np.float32), "B")],
            ),
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        quantizer = MatMulNBitsQuantizer(
            model,
            algo_config=DefaultWeightOnlyQuantConfig(is_symmetric=True, activation_quantization="dynamic_per_token"),
        )
        with self.assertRaisesRegex(ValueError, "known FP16/BF16 activation type"):
            quantizer.process()

    def test_rejects_incompatible_shapes_and_types(self):
        for invalid in ("output_type", "activation_k", "scalar_activation", "weight_type"):
            with self.subTest(invalid=invalid):
                model = helper.make_model(
                    helper.make_graph(
                        [helper.make_node("MatMul", ["A", "B"], ["Y"], name="projection")],
                        "invalid_export",
                        [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [2, 17])],
                        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 3])],
                        [numpy_helper.from_array(np.ones((17, 3), dtype=np.float16), "B")],
                    ),
                    opset_imports=[helper.make_opsetid("", 21)],
                    ir_version=10,
                )
                if invalid == "output_type":
                    model.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
                elif invalid == "activation_k":
                    model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value = 18
                elif invalid == "scalar_activation":
                    model.graph.input[0].type.tensor_type.shape.ClearField("dim")
                else:
                    model.graph.initializer[0].CopyFrom(
                        numpy_helper.from_array(np.ones((17, 3), dtype=np.float64), "B")
                    )
                quantizer = MatMulNBitsQuantizer(
                    model,
                    algo_config=DefaultWeightOnlyQuantConfig(
                        is_symmetric=True, activation_quantization="dynamic_per_token"
                    ),
                )
                with self.assertRaisesRegex(ValueError, "DynamicQuantizeMatMulNBits"):
                    quantizer.process()


if __name__ == "__main__":
    unittest.main()
