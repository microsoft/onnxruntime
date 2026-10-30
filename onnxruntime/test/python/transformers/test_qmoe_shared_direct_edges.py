# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CPU operator boundary tests for immutable INT4/INT3 expert banks."""

import unittest

import numpy as np
from onnx import TensorProto, helper
from test_qmoe_shared_direct import _fixture, _model, _reference, _round

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


if __name__ == "__main__":
    unittest.main()
