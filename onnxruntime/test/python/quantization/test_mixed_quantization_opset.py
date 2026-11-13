# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    get_qdq_config,
    quantize,
    quantize_static,
)
from onnxruntime.quantization.quant_utils import get_opset_version, update_opset_version


class _Reader(CalibrationDataReader):
    def __init__(self):
        self.data = iter(
            [
                {"X": np.array([[-2.0, -0.5, 1.0, 3.0], [0.25, 1.5, -1.0, 2.0]], dtype=np.float32)},
                {"X": np.array([[1.0, 2.0, -0.75, -1.5], [2.5, -1.0, 0.5, 0.75]], dtype=np.float32)},
            ]
        )

    def get_next(self):
        return next(self.data, None)


class TestMixedQuantizationOpset(unittest.TestCase):
    @staticmethod
    def _model(opset):
        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "W"], ["Y"], name="add")],
            "mixed-qdq",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
            [numpy_helper.from_array(np.array([-1.0, 0.5, 2.0, 3.0], dtype=np.float32), "W")],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)

    def _check_native_model(self, path, integer_type):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
        self.assertEqual(get_opset_version(model), 21)
        qdq_nodes = [node for node in model.graph.node if node.op_type in ("QuantizeLinear", "DequantizeLinear")]
        self.assertTrue(qdq_nodes)
        self.assertTrue(all(node.domain == "" for node in qdq_nodes))
        initializer_types = {initializer.data_type for initializer in model.graph.initializer}
        self.assertIn(TensorProto.FLOAT8E4M3FN, initializer_types)
        self.assertIn(integer_type.tensor_type, initializer_types)

        feed = {"X": np.array([[-1.0, 0.5, 1.0, 2.0], [0.25, -0.5, 1.5, -1.0]], dtype=np.float32)}
        actual = session.run(None, feed)[0]
        expected = ReferenceEvaluator(model).run(None, feed)[0]
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_float8_with_integer_override_uses_highest_required_opset(self):
        for integer_type in (QuantType.QInt16, QuantType.QUInt16):
            for nested in (False, True):
                override = {"quant_type": integer_type}
                if nested:
                    override = {"convert": override}
                for opset in (17, 18, 19, 20):
                    with self.subTest(integer_type=integer_type, nested=nested, opset=opset):
                        model = self._model(opset)
                        result = update_opset_version(
                            model,
                            QuantType.QFLOAT8E4M3FN,
                            QuantType.QFLOAT8E4M3FN,
                            tensor_quant_overrides={"X": [override]},
                        )
                        self.assertEqual(get_opset_version(result), 21)

    def test_direct_tensor_types_use_the_same_requirement(self):
        for integer_type in (TensorProto.INT16, TensorProto.UINT16):
            with self.subTest(integer_type=integer_type):
                result = update_opset_version(self._model(17), TensorProto.FLOAT8E4M3FN, integer_type)
                self.assertEqual(get_opset_version(result), 21)

    def test_public_quantize_static_produces_native_mixed_qdq(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.onnx"
            for opset in (17, 18):
                for integer_type in (QuantType.QInt16, QuantType.QUInt16):
                    for tensor in ("X", "W"):
                        with self.subTest(opset=opset, integer_type=integer_type, tensor=tensor):
                            quantize_static(
                                self._model(opset),
                                path,
                                _Reader(),
                                quant_format=QuantFormat.QDQ,
                                activation_type=QuantType.QFLOAT8E4M3FN,
                                weight_type=QuantType.QFLOAT8E4M3FN,
                                calibrate_method=CalibrationMethod.Distribution,
                                calibration_providers=["CPUExecutionProvider"],
                                extra_options={"TensorQuantOverrides": {tensor: [{"quant_type": integer_type}]}},
                            )
                            self._check_native_model(path, integer_type)

    def test_config_based_quantization_with_convert_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configured.onnx"
            for integer_type in (QuantType.QInt16, QuantType.QUInt16):
                with self.subTest(integer_type=integer_type):
                    model = self._model(17)
                    config = get_qdq_config(
                        model,
                        _Reader(),
                        calibrate_method=CalibrationMethod.Distribution,
                        activation_type=QuantType.QFLOAT8E4M3FN,
                        weight_type=QuantType.QFLOAT8E4M3FN,
                        tensor_quant_overrides={"X": [{"convert": {"quant_type": integer_type}}]},
                        calibration_providers=["CPUExecutionProvider"],
                    )
                    self.assertFalse(config.extra_options.get("UseQDQContribOps", False))
                    quantize(model, path, config)
                    self._check_native_model(path, integer_type)

    def test_other_version_rules_and_newer_models_are_preserved(self):
        cases = [
            (17, QuantType.QFLOAT8E4M3FN, QuantType.QFLOAT8E4M3FN, 0, 19),
            (19, QuantType.QFLOAT8E4M3FN, QuantType.QFLOAT8E4M3FN, 0, 19),
            (17, QuantType.QInt8, QuantType.QUInt8, 0, 17),
            (17, QuantType.QInt8, QuantType.QInt16, 0, 21),
            (17, QuantType.QFLOAT8E4M3FN, QuantType.QFLOAT8E4M3FN, 2, 21),
            (8, QuantType.QInt8, QuantType.QUInt8, 0, 11),
            (10, QuantType.QInt8, QuantType.QUInt8, 0, 10),
            (23, QuantType.QFLOAT8E4M3FN, QuantType.QInt16, 0, 23),
        ]
        for opset, weight_type, activation_type, block_size, expected in cases:
            with self.subTest(
                opset=opset, weight_type=weight_type, activation_type=activation_type, block_size=block_size
            ):
                model = self._model(opset)
                result = update_opset_version(model, weight_type, activation_type, block_size=block_size)
                self.assertEqual(get_opset_version(result), expected)
                if expected == opset:
                    self.assertIs(result, model)

    def test_upgrade_does_not_mutate_the_supplied_model(self):
        model = self._model(17)
        before = model.SerializeToString()
        result = update_opset_version(
            model,
            QuantType.QFLOAT8E4M3FN,
            QuantType.QFLOAT8E4M3FN,
            tensor_quant_overrides={"W": [{"quant_type": QuantType.QInt16}]},
        )
        self.assertEqual(get_opset_version(result), 21)
        self.assertEqual(model.SerializeToString(), before)
        self.assertIsNot(result, model)


if __name__ == "__main__":
    unittest.main()
