# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CPU operator boundary tests for immutable INT4/INT3 expert banks."""

import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from test_qmoe_shared_direct import _fixture, _model, _pack_int3, _pack_sm80, _reference, _round

import onnxruntime as ort


def _typed_tensor(name, values, dtype):
    values = _round(values, dtype)
    if dtype == TensorProto.BFLOAT16:
        raw = (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    else:
        raw = values.astype(np.float16).tobytes()
    return helper.make_tensor(name, dtype, values.shape, raw, raw=True)


def _node(model):
    return next(node for node in model.graph.node if node.op_type == "QMoE")


def _set_attribute(node, name, value):
    retained = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(retained)
    node.attribute.append(helper.make_attribute(name, value))


def _set_input(node, slot, name):
    while len(node.input) <= slot:
        node.input.append("")
    node.input[slot] = name


def _optional_reference(fixture, x, router, bias1=None, bias2=None, router_weights=None, normalize=True):
    dtype = fixture["dtype"]
    if bias1 is None and bias2 is None and router_weights is None:
        return _reference(fixture, x, router, normalize=normalize)
    x, router = _round(x, dtype), _round(router, dtype)
    bias1 = None if bias1 is None else _round(bias1, dtype)
    bias2 = None if bias2 is None else _round(bias2, dtype)
    router_weights = None if router_weights is None else _round(router_weights, dtype)
    matrices = [
        _round(codes * np.repeat(scales, fixture["block"], axis=-1), dtype).astype(np.float64)
        for codes, scales in zip(fixture["codes"], fixture["scales"], strict=True)
    ]
    output = np.zeros_like(x, dtype=np.float64)
    for row in range(x.shape[0]):
        selected = np.argsort(-router[row], kind="stable")[:2]
        probabilities = np.exp(router[row].astype(np.float64) - float(router[row].max()))
        denominator = probabilities[selected].sum() if normalize else probabilities.sum()
        for expert in selected:
            fc1 = x[row].astype(np.float64) @ matrices[0][expert].T
            if bias1 is not None:
                fc1 += bias1[expert]
            gate, up = fc1[0::2], fc1[1::2]
            intermediate = _round(gate / (1 + np.exp(-gate)) * up, dtype)
            fc2 = intermediate.astype(np.float64) @ matrices[1][expert].T
            if bias2 is not None:
                fc2 += bias2[expert]
            fc2 = _round(fc2, dtype)
            scale = probabilities[expert] / denominator if router_weights is None else router_weights[row, expert]
            output[row] += fc2 * scale
    return _round(output, dtype)


class TestQMoESharedDirectEdges(unittest.TestCase):
    def _session(self, model, disable_prepacking=False):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_prepacking:
            options.add_session_config_entry("session.disable_prepacking", "1")
        return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])

    def _inputs(self, fixture, rows=5):
        rng = np.random.default_rng(3405)
        experts, _, hidden = fixture["codes"][0].shape
        return {
            "x": rng.uniform(-0.4, 0.4, (rows, hidden)).astype(np.float32),
            "router": rng.normal(size=(rows, experts)).astype(np.float32),
        }

    def test_fc1_and_fc2_biases_observe_activation_boundaries(self):
        for bits in [3, 4]:
            for dtype in [TensorProto.FLOAT16, TensorProto.BFLOAT16]:
                fixture = _fixture(bits, dtype)
                feeds = self._inputs(fixture)
                rng = np.random.default_rng(230)
                for use_fc1, use_fc2 in [(True, False), (False, True), (True, True)]:
                    with self.subTest(bits=bits, dtype=dtype, fc1=use_fc1, fc2=use_fc2):
                        model, _ = _model(fixture)
                        node = _node(model)
                        bias1 = rng.uniform(-0.15, 0.15, fixture["codes"][0].shape[:2]).astype(np.float32)
                        bias2 = rng.uniform(-0.1, 0.1, fixture["codes"][1].shape[:2]).astype(np.float32)
                        if use_fc1:
                            model.graph.initializer.append(_typed_tensor("bias1", bias1, dtype))
                            _set_input(node, 4, "bias1")
                        if use_fc2:
                            model.graph.initializer.append(_typed_tensor("bias2", bias2, dtype))
                            _set_input(node, 7, "bias2")
                        expected = _optional_reference(
                            fixture,
                            feeds["x"],
                            feeds["router"],
                            bias1=bias1 if use_fc1 else None,
                            bias2=bias2 if use_fc2 else None,
                        )
                        for disable_prepacking in [False, True]:
                            actual = self._session(model, disable_prepacking).run(None, feeds)[0]
                            np.testing.assert_allclose(actual, expected, rtol=0.025, atol=0.0016)

    def test_router_weights_override_is_used_without_renormalization(self):
        for bits in [3, 4]:
            for dtype in [TensorProto.FLOAT16, TensorProto.BFLOAT16]:
                fixture = _fixture(bits, dtype)
                feeds = self._inputs(fixture)
                weights = np.tile(np.array([-0.75, 2.5, 0.125, 1.25], np.float32), (feeds["x"].shape[0], 1))
                outputs = []
                for normalize in [False, True]:
                    with self.subTest(bits=bits, dtype=dtype, normalize=normalize):
                        model, _ = _model(fixture, normalize=normalize)
                        model.graph.initializer.append(_typed_tensor("mixing_weights", weights, dtype))
                        _set_input(_node(model), 14, "mixing_weights")
                        expected = _optional_reference(
                            fixture,
                            feeds["x"],
                            feeds["router"],
                            router_weights=weights,
                            normalize=normalize,
                        )
                        actual = self._session(model).run(None, feeds)[0]
                        np.testing.assert_allclose(actual, expected, rtol=0.025, atol=0.0016)
                        outputs.append(actual)
                np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_auxiliary_scales_are_rejected(self):
        for bits in [3, 4]:
            for slot in [15, 16, 17, 18]:
                with self.subTest(bits=bits, slot=slot):
                    fixture = _fixture(bits, TensorProto.BFLOAT16)
                    model, _ = _model(fixture)
                    name = f"unsupported_scale_{slot}"
                    model.graph.initializer.append(helper.make_tensor(name, TensorProto.FLOAT, [1], [1.0]))
                    _set_input(_node(model), slot, name)
                    with self.assertRaisesRegex(Exception, "auxiliary scales|does not accept input"):
                        self._session(model)

    def test_k_outside_expert_count_is_rejected(self):
        for bits in [3, 4]:
            fixture = _fixture(bits, TensorProto.FLOAT16)
            for top_k in [0, 5]:
                with self.subTest(bits=bits, top_k=top_k):
                    model, _ = _model(fixture)
                    _set_attribute(_node(model), "k", top_k)
                    with self.assertRaisesRegex(Exception, "[Kk]|top_k|expert"):
                        self._session(model).run(None, self._inputs(fixture))

    def test_invalid_dynamic_scale_grid_is_rejected(self):
        for bits in [3, 4]:
            fixture = _fixture(bits, TensorProto.FLOAT16)
            model, feeds = _model(fixture, dynamic=True)
            scale_name = "direct_qmoe_s0_float"
            original = feeds[scale_name]
            feeds[scale_name] = original.reshape(original.shape[0], original.shape[1] // 2, original.shape[2] * 2)
            scale_input = next(value for value in model.graph.input if value.name == scale_name)
            for axis, size in zip(scale_input.type.tensor_type.shape.dim, feeds[scale_name].shape, strict=True):
                axis.dim_value = size
            feeds.update(self._inputs(fixture))
            with self.subTest(bits=bits), self.assertRaisesRegex(Exception, "scale|dimension|shape"):
                self._session(model).run(None, feeds)

    def test_empty_rows_preserve_output_shape(self):
        for bits in [3, 4]:
            for dtype in [TensorProto.FLOAT16, TensorProto.BFLOAT16]:
                with self.subTest(bits=bits, dtype=dtype):
                    fixture = _fixture(bits, dtype)
                    model, _ = _model(fixture)
                    feeds = self._inputs(fixture, rows=0)
                    actual = self._session(model).run(None, feeds)[0]
                    self.assertEqual(actual.shape, feeds["x"].shape)
                    self.assertEqual(actual.size, 0)


class TestQMoEReviewRegressions(unittest.TestCase):
    def _run_int3(self, fixture, x, bias):
        fixture["packed"] = [_pack_int3(codes) for codes in fixture["codes"]]
        model, feeds = _model(fixture, top_k=1)
        model.graph.initializer.append(_typed_tensor("bias", bias, fixture["dtype"]))
        _set_input(_node(model), 4, "bias")
        feeds.update(x=x, router=np.zeros((len(x), 1), dtype=np.float32))
        return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)[0]

    def test_int3_rounds_each_weight_before_accumulation(self):
        fixture = _fixture(3, TensorProto.BFLOAT16, hidden=64, inter=64, experts=1)
        for codes in fixture["codes"]:
            codes.fill(0)
        for scales in fixture["scales"]:
            scales.fill(1)
        fixture["codes"][0][0, 1, :2] = [3, 1]
        fixture["scales"][0][0, 1, 0] = np.array([0x3DCD0000], dtype=np.uint32).view(np.float32)[0]
        fixture["codes"][1][0, 0, 0] = 1
        x = np.zeros((1, 64), dtype=np.float32)
        x[0, :2] = [1024, -3072]
        bias = np.zeros((1, 128), dtype=np.float32)
        bias[0, 0] = 1
        output = self._run_int3(fixture, x, bias)
        self.assertEqual(output[0, 0], 0.365234375)

    def test_int3_fp16_weight_rounding_before_cancellation(self):
        fixture = _fixture(3, TensorProto.FLOAT16, hidden=64, inter=64, experts=1)
        for codes in fixture["codes"]:
            codes.fill(0)
        for scales in fixture["scales"]:
            scales.fill(1)
        fixture["codes"][0][0, 1, :2] = [3, 1]
        fixture["scales"][0][0, 1, 0] = np.float16(0.1)
        fixture["codes"][1][0, 0, 0] = 1
        x = np.zeros((1, 64), dtype=np.float32)
        x[0, :2] = [1024, -3072]
        bias = np.zeros((1, 128), dtype=np.float32)
        bias[0, 0] = 1
        expected = np.float16(-0.125 / (1 + np.exp(-1))).astype(np.float32)
        self.assertEqual(self._run_int3(fixture, x, bias)[0, 0], expected)

    def test_int3_preserves_float_projections_until_swiglu(self):
        fixture = _fixture(3, TensorProto.FLOAT16, hidden=64, inter=64, experts=1)
        for codes in fixture["codes"]:
            codes.fill(0)
        for scales in fixture["scales"]:
            scales.fill(1)
        fixture["codes"][0][0, 0, :2] = [1, 1]
        fixture["codes"][0][0, 2, 0] = 1
        fixture["codes"][1][0, 0, :2] = [2, -2]
        fixture["scales"][1][0, 0, 0] = 512
        x = np.zeros((1, 64), dtype=np.float32)
        x[0, :2] = [1, 2**-11]
        bias = np.zeros((1, 128), dtype=np.float32)
        bias[0, [1, 3]] = 1
        output = self._run_int3(fixture, x, bias)
        self.assertEqual(output[0, 0], 0.5)

    @staticmethod
    def _fp4_model(kind, hidden, inter, bias1, bias2):
        block = 16 if kind == "nvfp4" else 32
        scale_type = TensorProto.FLOAT8E4M3FN if kind == "nvfp4" else TensorProto.FLOAT8E8M0
        scale_value = 0x38 if kind == "nvfp4" else 0x7F
        w1 = np.zeros((1, 2 * inter, hidden // 2), dtype=np.uint8)
        w2 = np.zeros((1, hidden, inter // 2), dtype=np.uint8)
        w2[0, 0, 0] = 2  # Select the first activated value with weight 1.
        initializers = [numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(w2, "w2")]
        for name, shape in [("s1", (1, 2 * inter, hidden // block)), ("s2", (1, hidden, inter // block))]:
            initializers.append(
                helper.make_tensor(name, scale_type, shape, bytes([scale_value]) * np.prod(shape), raw=True)
            )
        initializers.extend([numpy_helper.from_array(bias1, "b1"), numpy_helper.from_array(bias2, "b2")])
        initializers.extend([numpy_helper.from_array(np.ones(1, np.float32), name) for name in ("g1", "g2")])
        inputs = ["x", "router", "w1", "s1", "b1", "w2", "s2", "b2"] + [""] * 7 + ["g1", "g2"]
        node = helper.make_node(
            "QMoE",
            inputs,
            ["output"],
            domain="com.microsoft",
            k=1,
            activation_type="swiglu",
            swiglu_fusion=1,
            expert_weight_bits=4,
            quant_type=kind,
            weights_prepacked=1,
            block_size=block,
        )
        graph = helper.make_graph(
            [node],
            "fp4_review",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, hidden]),
                helper.make_tensor_value_info("router", TensorProto.FLOAT, [1, 1]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, hidden])],
            initializers,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 10
        return model

    def test_fp4_float_biases(self):
        for kind in ("nvfp4", "fp4"):
            for use_fc1, use_fc2 in ((True, False), (False, True), (True, True)):
                with self.subTest(kind=kind, fc1=use_fc1, fc2=use_fc2):
                    b1 = np.zeros((1, 64), np.float32)
                    b2 = np.zeros((1, 32), np.float32)
                    if use_fc1:
                        b1[0, :2] = 1
                    if use_fc2:
                        b2.fill(1)
                    model = self._fp4_model(kind, 32, 32, b1, b2)
                    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
                    actual = session.run(
                        None, {"x": np.zeros((1, 32), np.float32), "router": np.zeros((1, 1), np.float32)}
                    )[0]
                    expected = b2.copy()
                    if use_fc1:
                        expected[0, 0] += 1 / (1 + np.exp(-1))
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_nvfp4_rejects_partial_32_element_tiles(self):
        for hidden, inter in ((16, 16), (48, 32), (32, 48), (32, 32)):
            with self.subTest(hidden=hidden, inter=inter):
                model = self._fp4_model(
                    "nvfp4", hidden, inter, np.zeros((1, 2 * inter), np.float32), np.zeros((1, hidden), np.float32)
                )

                def run(model=model, hidden=hidden):
                    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
                    return session.run(
                        None, {"x": np.zeros((1, hidden), np.float32), "router": np.zeros((1, 1), np.float32)}
                    )

                if hidden % 32 or inter % 32:
                    with self.assertRaisesRegex(Exception, "divisible by 32"):
                        run()
                else:
                    self.assertEqual(run()[0].shape, (1, hidden))


class TestQMoEDeferredScale(unittest.TestCase):
    @staticmethod
    def _session(model, mode=None):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if mode is not None:
            options.add_session_config_entry("ep.cpu.qmoe_deferred_scale", mode)
        return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])

    def test_fc1_and_fc2_cancellation_in_both_modes(self):
        for bits in (3, 4):
            for dtype in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
                for block in (32, 64, 128):
                    for fc2 in (False, True):
                        with self.subTest(bits=bits, dtype=dtype, block=block, fc2=fc2):
                            fixture = _fixture(bits, dtype, hidden=128, inter=128, experts=1)
                            fixture["block"] = block
                            fixture["scales"] = [np.ones((1, n, 128 // block), np.float32) for n in (256, 128)]
                            for codes in fixture["codes"]:
                                codes.fill(0)
                            scale = (
                                np.array([0x3DCD0000], np.uint32).view(np.float32)[0]
                                if dtype == TensorProto.BFLOAT16
                                else np.float32(np.float16(0.1))
                            )
                            projection, column = (1, 0) if fc2 else (0, 1)
                            fixture["codes"][projection][0, column, :2] = [3, 1]
                            fixture["scales"][projection][0, column, 0] = scale
                            bias = np.zeros((1, 256), np.float32)
                            x = np.zeros((9, 128), np.float32)
                            if fc2:
                                # alpha=0 and gate=2 make the SwiGLU factor exactly 1.
                                bias[0, [0, 2]] = 2
                                bias[0, [1, 3]] = [1024, -3072]
                            else:
                                bias[0, 0] = 1
                                x[:, :2] = [1024, -3072]
                                fixture["codes"][1][0, 0, 0] = 1
                            pack = _pack_int3 if bits == 3 else _pack_sm80
                            fixture["packed"] = [pack(codes) for codes in fixture["codes"]]
                            model, feeds = _model(fixture, top_k=1)
                            model.graph.initializer.append(_typed_tensor("bias", bias, dtype))
                            node = _node(model)
                            _set_input(node, 4, "bias")
                            if fc2:
                                _set_attribute(node, "activation_alpha", 0.0)
                            feeds.update(x=x, router=np.zeros((len(x), 1), np.float32))
                            typed = float(_round(3 * scale, dtype)) * 1024 - float(scale) * 3072
                            expected = np.zeros_like(x)
                            expected[:, 0] = _round(typed if fc2 else typed / (1 + np.exp(-1)), dtype)
                            sessions = {mode: self._session(model, mode) for mode in (None, "0", "1")}
                            # Alternate independent sessions to catch process-global policy state.
                            for mode in ("0", "1", None, "1", "0"):
                                actual = sessions[mode].run(None, feeds)[0]
                                np.testing.assert_array_equal(actual, np.zeros_like(x) if mode == "1" else expected)

    def test_invalid_values_are_rejected(self):
        model, _ = _model(_fixture(3, TensorProto.FLOAT16))
        for value in ("", "2", "-1", "true"):
            with self.subTest(value=value), self.assertRaisesRegex(Exception, "ep.cpu.qmoe_deferred_scale.*must be"):
                self._session(model, value)

    def test_fp4_is_unaffected(self):
        for kind in ("nvfp4", "fp4"):
            with self.subTest(kind=kind):
                model = TestQMoEReviewRegressions._fp4_model(
                    kind, 32, 32, np.ones((1, 64), np.float32), np.zeros((1, 32), np.float32)
                )
                for initializer in model.graph.initializer:
                    if initializer.name in ("w1", "w2"):
                        values = np.full(tuple(initializer.dims), 0x23, np.uint8)
                        initializer.CopyFrom(numpy_helper.from_array(values, initializer.name))
                    elif initializer.name in ("g1", "g2"):
                        initializer.CopyFrom(numpy_helper.from_array(np.array([0.1], np.float32), initializer.name))
                feeds = {
                    "x": np.linspace(-0.5, 0.5, 32, dtype=np.float32)[None],
                    "router": np.zeros((1, 1), np.float32),
                }
                expected = self._session(model).run(None, feeds)[0]
                self.assertGreater(float(np.max(np.abs(expected))), 0)
                for mode in ("0", "1"):
                    np.testing.assert_array_equal(self._session(model, mode).run(None, feeds)[0], expected)


if __name__ == "__main__":
    unittest.main()
