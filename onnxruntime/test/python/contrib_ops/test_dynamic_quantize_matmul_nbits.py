# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Session-level coverage for the generic CUDA DynamicQuantizeMatMulNBits implementation."""

import unittest
from concurrent.futures import ThreadPoolExecutor

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


def reference(a, packed, scales, bits, block, mode, activation_block):
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
        acc = quantized[:, start:end] @ weights[:, start:end].T
        term = (acc.astype(np.float32) * a_scales[:, start, None]) * scales[:, start // block].astype(np.float32)
        output = term if output is None else output + term
        start = end
    return (output + np.float32(0)).astype(np.float16)


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

    def test_grouped_packed_weights_match_exact_reference(self):
        rng = np.random.default_rng(0)
        block = 128
        activation_block = 192
        for m, k, n in ((65, 320, 137), (961, 192, 961)):
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

    def test_capture_replays_changed_activations_and_weights(self):
        for m, k, n, block in ((2, 17, 3, 16), (33, 64, 37, 128), (129, 272, 137, 512), (65, 320, 137, 128)):
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

    def test_concurrent_runs_use_independent_activation_scratch(self):
        for m, k, n, block in ((2, 17, 3, 16), (33, 64, 37, 128), (129, 272, 137, 512), (65, 320, 137, 128)):
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
