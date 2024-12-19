#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnx
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count

from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, get_qdq_config, quantize


class TestGetQDQConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.int_qdq_config_")

        # Note: swap with the commented line if you want to see the models in local test dir.
        cls._tmp_dir_path = cls._tmp_model_dir.name
        # cls._tmp_dir_path = "."

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def build_add_model(
        self,
        shape: list[int],
        tensor_type: onnx.TensorProto.DataType,
        weight: onnx.TensorProto | None = None,
        opset: int = 21,
    ) -> onnx.ModelProto:
        """
        Returns an onnx.ModelProto with a single Add operator. The second input can be optionally made
        a static weight.
        """
        graph_inputs = [onnx.helper.make_tensor_value_info("input_0", tensor_type, shape)]
        graph_outputs = [onnx.helper.make_tensor_value_info("output_0", tensor_type, shape)]
        initializers = []
        add_input_names = ["input_0"]

        if weight is not None:
            initializers.append(weight)
            add_input_names.append(weight.name)
        else:
            graph_inputs.append(onnx.helper.make_tensor_value_info("input_1", tensor_type, shape))
            add_input_names.append("input_1")

        add_node = onnx.helper.make_node("Add", add_input_names, ["output_0"], name="Add0")

        graph = onnx.helper.make_graph(
            [add_node],
            "AddGraph",
            graph_inputs,
            graph_outputs,
            initializer=initializers,
        )
        opset_imports = [onnx.helper.make_opsetid("", opset)]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, True)
        return model

    def test_basic_args(self):
        """
        Test that get_qdq_config() returns a config that sets the basic args.
        """

        shape = [1, 8, 8]
        tensor_type = onnx.TensorProto.FLOAT
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type)
        weight = onnx.numpy_helper.from_array(np.ones(shape, dtype=np_dtype), "weight")
        float_model = self.build_add_model(shape, tensor_type, weight, opset=21)

        input_data_list = [
            {"input_0": np.ones(shape, dtype=np_dtype) * np.array(-2, dtype=np_dtype)},
            {"input_0": np.ones(shape, dtype=np_dtype) * np.array(2, dtype=np_dtype)},
        ]
        data_reader = TestDataFeeds(input_data_list)

        qdq_config = get_qdq_config(
            float_model,
            data_reader,
            calibrate_method=CalibrationMethod.Percentile,
            calibrate_args={"percentile": 99.98},  # Converted to extra_options
            activation_type=QuantType.QUInt16,
            weight_type=QuantType.QInt16,
            per_channel=True,
            reduce_range=True,
            nodes_to_exclude=["Mul"],
            # Other options converted to extra_options:
            min_real_range=0.0001,
            keep_removable_activations=True,
            activation_symmetric=True,
            weight_symmetric=True,
        )
        self.assertEqual(qdq_config.calibrate_method, CalibrationMethod.Percentile)
        self.assertEqual(qdq_config.activation_type, QuantType.QUInt16)
        self.assertEqual(qdq_config.weight_type, QuantType.QInt16)
        self.assertTrue(qdq_config.per_channel)
        self.assertTrue(qdq_config.reduce_range)
        self.assertEqual(set(qdq_config.nodes_to_exclude), {"Mul"})
        self.assertEqual(set(qdq_config.op_types_to_quantize), {"Add"})

        # Check that calibration args are translated to extra_options.
        self.assertEqual(qdq_config.extra_options["CalibPercentile"], 99.98)

        # Check that other args are also translated to extra_options.
        self.assertEqual(qdq_config.extra_options["MinimumRealRange"], 0.0001)
        self.assertTrue(qdq_config.extra_options["QDQKeepRemovableActivations"])
        self.assertTrue(qdq_config.extra_options["ActivationSymmetric"])
        self.assertTrue(qdq_config.extra_options["WeightSymmetric"])

        # The following options should always be set to specific values.
        self.assertTrue(qdq_config.extra_options["ForceQuantizeNoInputCheck"])
        self.assertEqual(qdq_config.quant_format, QuantFormat.QDQ)

        # Should use onnx domain Q/DQ ops because onnx opset >= 21.
        self.assertFalse(qdq_config.extra_options.get("UseQDQContribOps", False))

    def test_exclude_nodes_callable(self):
        """
        Test passing a function/callable to exclude nodes from quantization.
        """

        shape = [1, 8, 8]
        tensor_type = onnx.TensorProto.FLOAT
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type)
        weight = onnx.numpy_helper.from_array(np.ones(shape, dtype=np_dtype), "weight")
        float_model = self.build_add_model(shape, tensor_type, weight, opset=21)

        input_data_list = [
            {"input_0": np.ones(shape, dtype=np_dtype) * np.array(-2, dtype=np_dtype)},
            {"input_0": np.ones(shape, dtype=np_dtype) * np.array(2, dtype=np_dtype)},
        ]
        data_reader = TestDataFeeds(input_data_list)

        # Local function that excludes all "Add" nodes.
        def should_exclude_node_(model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
            return node.op_type == "Add"

        qdq_config = get_qdq_config(
            float_model,
            data_reader,
            nodes_to_exclude=should_exclude_node_,
        )

        expected_excluded_nodes = set([node.name for node in float_model.graph.node if node.op_type == "Add"])
        self.assertTrue(bool(expected_excluded_nodes))
        self.assertEqual(set(qdq_config.nodes_to_exclude), expected_excluded_nodes)

    def test_external_data(self):
        """
        Test that get_qdq_config() returns a config that enables external data
        if the input model has external data.
        """

        # Create model with a weight large enough (> 1024 bytes) to be stored externally.
        shape = [1, 32, 32]
        tensor_type = onnx.TensorProto.FLOAT
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type)
        large_weight = onnx.numpy_helper.from_array(np.ones(shape, dtype=np_dtype), "weight")
        float_model = self.build_add_model(shape, tensor_type, large_weight)
        float_model_path = os.path.join(self._tmp_dir_path, "add_ext_data_int_qdq_config.onnx")

        onnx.save_model(
            float_model,
            float_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="add_ext_data_int_qdq_config.bin",
        )

        input_data_list = [
            {"input_0": np.ones(shape, dtype=np_dtype) * np.array(-2, dtype=np_dtype)},
            {"input_0": np.ones(shape, dtype=np_dtype) * np.array(0, dtype=np_dtype)},
            {"input_0": np.ones(shape, dtype=np_dtype) * np.array(2, dtype=np_dtype)},
        ]
        data_reader = TestDataFeeds(input_data_list)

        # Create a quantization config and check that it sets boolean to use external data
        qdq_config = get_qdq_config(
            float_model_path, data_reader, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8
        )
        self.assertEqual(set(qdq_config.op_types_to_quantize), {"Add"})
        self.assertTrue(qdq_config.use_external_data_format)

        # Quantize the model and check computational correctness against float model.
        qdq_model_path = os.path.join(self._tmp_dir_path, "add_ext_data_int_qdq_config.qdq.onnx")
        quantize(float_model_path, qdq_model_path, qdq_config)

        expected_op_counts = {"DequantizeLinear": 3, "QuantizeLinear": 2, "Add": 1}
        check_op_type_count(self, qdq_model_path, **expected_op_counts)

        data_reader.rewind()
        check_model_correctness(self, float_model_path, qdq_model_path, data_reader.get_next())

        # The quantized weight should still be stored in an external file.
        qdq_model = onnx.load_model(qdq_model_path, load_external_data=False)
        weight_quantized = next(
            (
                initializer
                for initializer in qdq_model.graph.initializer
                if initializer.name == f"{large_weight.name}_quantized"
            ),
            None,
        )
        self.assertIsNotNone(weight_quantized)
        self.assertEqual(weight_quantized.data_location, onnx.TensorProto.EXTERNAL)

    def test_use_qdq_contrib_ops_for_int16_opset19(self):
        """
        Test that get_qdq_config() returns a config that forces 'com.microsoft' Q/DQ ops for
        use of int16 in opset < 21.
        """

        shape = [1, 8, 8]
        tensor_type = onnx.TensorProto.FLOAT
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type)
        weight = onnx.numpy_helper.from_array(np.ones(shape, dtype=np_dtype), "weight")
        float_model = self.build_add_model(shape, tensor_type, weight, opset=19)

        input_data_list = [
            {"input_0": np.ones(shape, dtype=np_dtype) * np.array(-2, dtype=np_dtype)},
            {"input_0": np.ones(shape, dtype=np_dtype) * np.array(2, dtype=np_dtype)},
        ]
        data_reader = TestDataFeeds(input_data_list)

        qdq_config = get_qdq_config(
            float_model,
            data_reader,
            activation_type=QuantType.QUInt16,
            weight_type=QuantType.QInt8,
        )

        self.assertEqual(qdq_config.activation_type, QuantType.QUInt16)
        self.assertTrue(qdq_config.extra_options["UseQDQContribOps"])

    def test_use_qdq_contrib_ops_for_int4_opset19(self):
        """
        Test that get_qdq_config() returns a config that forces 'com.microsoft' Q/DQ ops for
        use of int4 in opset < 21.
        """

        shape = [1, 8, 8]
        tensor_type = onnx.TensorProto.FLOAT
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type)
        weight = onnx.numpy_helper.from_array(np.ones(shape, dtype=np_dtype), "weight")
        float_model = self.build_add_model(shape, tensor_type, weight, opset=19)

        input_data_list = [
            {"input_0": np.ones(shape, dtype=np_dtype) * np.array(-2, dtype=np_dtype)},
            {"input_0": np.ones(shape, dtype=np_dtype) * np.array(2, dtype=np_dtype)},
        ]
        data_reader = TestDataFeeds(input_data_list)

        # Use int4 in tensor quantization overrides. This should still force use of 'com.microsoft' Q/DQ ops.
        qdq_config = get_qdq_config(
            float_model,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            tensor_quant_overrides={"weight": [{"quant_type": QuantType.QInt4}]},
        )

        self.assertEqual(qdq_config.extra_options["TensorQuantOverrides"]["weight"][0]["quant_type"], QuantType.QInt4)
        self.assertTrue(qdq_config.extra_options["UseQDQContribOps"])


if __name__ == "__main__":
    unittest.main()
