# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Tests for the save_activations module."""

import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper
from op_test_utils import generate_random_initializer

import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.qdq_loss_debug import (
    QUANT_INPUT_SUFFIX,
    collect_activations,
    compute_activation_error,
    compute_weight_error,
    create_activation_matching,
    create_weight_matching,
    modify_model_output_intermediate_tensors,
)


def construct_test_model1(test_model_path: str, activations_as_outputs=False):
    """ Create an ONNX model shaped as:
    ```
       (input)
          |
         Relu1
         /   \
      Conv1   \
        |      \
      Relu2  Conv3
        |      |
      Conv2    |
        \\      /
          Add
           |
          (AddOut)
    ```
    We are keeping all intermediate tensors as output, just for test verification
    purposes
    """

    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1, 3])
    x1_output = helper.make_tensor_value_info("Relu1Out", TensorProto.FLOAT, [1, 3, 1, 3])
    x2_output = helper.make_tensor_value_info("Conv1Out", TensorProto.FLOAT, [1, 3, 1, 3])
    x3_output = helper.make_tensor_value_info("Relu2Out", TensorProto.FLOAT, [1, 3, 1, 3])
    x4_output = helper.make_tensor_value_info("Conv2Out", TensorProto.FLOAT, [1, 3, 1, 3])
    x5_output = helper.make_tensor_value_info("Conv3Out", TensorProto.FLOAT, [1, 3, 1, 3])
    x6_output = helper.make_tensor_value_info("AddOut", TensorProto.FLOAT, [1, 3, 1, 3])

    initializer = []
    initializer.append(generate_random_initializer("W1", [3, 3, 1, 1], np.float32))
    initializer.append(generate_random_initializer("B1", [3], np.float32))
    initializer.append(generate_random_initializer("W3", [3, 3, 1, 1], np.float32))
    initializer.append(generate_random_initializer("B3", [3], np.float32))
    initializer.append(generate_random_initializer("W5", [3, 3, 1, 1], np.float32))
    initializer.append(generate_random_initializer("B5", [3], np.float32))

    nodes = []
    nodes.append(helper.make_node("Relu", ["input"], ["Relu1Out"], name="Relu1"))
    nodes.append(helper.make_node("Conv", ["Relu1Out", "W1", "B1"], ["Conv1Out"], name="Conv1"))
    nodes.append(helper.make_node("Relu", ["Conv1Out"], ["Relu2Out"], name="Relu2"))
    nodes.append(helper.make_node("Conv", ["Relu2Out", "W3", "B3"], ["Conv2Out"], name="Conv2"))
    nodes.append(helper.make_node("Conv", ["Relu1Out", "W5", "B5"], ["Conv3Out"], name="Conv3"))
    nodes.append(helper.make_node("Add", ["Conv2Out", "Conv3Out"], ["AddOut"], name="Add"))

    # we are keeping all tensors in the output anyway for verification purpose
    outputs = [x6_output]
    if activations_as_outputs:
        outputs.extend([x1_output, x2_output, x3_output, x4_output, x5_output])
    graph = helper.make_graph(nodes, "test_graph_relu_conv", [input_vi], outputs, initializer=initializer)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, test_model_path)


class TestDataReader(CalibrationDataReader):
    """Random Data Input Generator"""

    def __init__(self, input_shape=[1, 3, 1, 3]):  # noqa: B006
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.count = 2
        self.input_data_list = []
        for _ in range(self.count):
            self.input_data_list.append(np.random.normal(0, 0.33, input_shape).astype(np.float32))

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            input_name = "input"
            self.enum_data_dicts = iter([{input_name: input_data} for input_data in self.input_data_list])
        return next(self.enum_data_dicts, None)

    def rewind(self):
        self.preprocess_flag = True


def augment_model_collect_activations(
    model_path: str, augmented_model_path: str, data_reader: TestDataReader
) -> Dict[str, List[np.ndarray]]:
    modify_model_output_intermediate_tensors(model_path, augmented_model_path)

    tensor_dict = collect_activations(augmented_model_path, data_reader)
    return tensor_dict


class TestSaveActivations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_save_activations.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def test_saved_tensors_match_internal_tensors(self):
        test_model_path = str(Path(self._tmp_model_dir.name) / "test_model1.onnx")
        construct_test_model1(test_model_path, activations_as_outputs=True)
        data_reader = TestDataReader()

        augmented_model_path = str(Path(self._tmp_model_dir.name).joinpath("augmented_test_model_1.onnx"))
        tensor_dict = augment_model_collect_activations(test_model_path, augmented_model_path, data_reader)

        # run original model and compare the tensors
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        infer_session = onnxruntime.InferenceSession(
            test_model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        data_reader.rewind()
        oracle_outputs = []
        for input_d in data_reader:
            oracle_outputs.append(infer_session.run(None, input_d))

        output_dict = {}
        output_info = infer_session.get_outputs()
        for batch in oracle_outputs:
            for output, output_data in zip(output_info, batch):
                output_dict.setdefault(output.name, []).append(output_data)

        for output_name, model_outputs in output_dict.items():
            test_outputs = tensor_dict[output_name]
            for expected, actual in zip(model_outputs, test_outputs):
                exp = expected.reshape(-1)
                act = actual.reshape(-1)
                np.testing.assert_equal(exp, act)

    def test_create_activation_matching_present(self):
        float_model_path = str(Path(self._tmp_model_dir.name) / "float_model2.onnx")
        construct_test_model1(float_model_path, activations_as_outputs=False)
        data_reader = TestDataReader()

        qdq_model_path = str(Path(self._tmp_model_dir.name) / "qdq_model2.onnx")
        quantize_static(
            float_model_path,
            qdq_model_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=False,
            reduce_range=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )

        data_reader.rewind()
        augmented_float_model_path = str(Path(self._tmp_model_dir.name).joinpath("augmented_float_model2.onnx"))
        float_activations = augment_model_collect_activations(float_model_path, augmented_float_model_path, data_reader)

        data_reader.rewind()
        augmented_qdq_model_path = str(Path(self._tmp_model_dir.name).joinpath("augmented_qdq_model2.onnx"))
        qdq_activations = augment_model_collect_activations(qdq_model_path, augmented_qdq_model_path, data_reader)

        compare_dict = create_activation_matching(qdq_activations, float_activations)

        # 'Conv1Out' is combined with 'Relu2Out'
        tensor_names = [
            "Relu1Out",
            "Relu2Out",
            "Conv2Out",
            "Conv3Out",
            "AddOut",
        ]
        for tensor_name in tensor_names:
            self.assertTrue(compare_dict[tensor_name]["float"])
            self.assertTrue(compare_dict[tensor_name]["pre_qdq"])
            self.assertTrue(compare_dict[tensor_name]["post_qdq"])

        self.assertFalse(compare_dict.get("Conv1Out"))

        activations_error = compute_activation_error(compare_dict)
        for tensor_name in tensor_names:
            self.assertIsInstance(
                activations_error[tensor_name]["xmodel_err"],
                float,
                f"{tensor_name} cross model error {activations_error[tensor_name]['xmodel_err']} not found!",
            )
            self.assertIsInstance(
                activations_error[tensor_name]["qdq_err"],
                float,
                f"{tensor_name} qdq error {activations_error[tensor_name]['qdq_err']} exceeds threashold.",
            )

    def test_create_weight_matching(self):
        # Setup: create float model:
        float_model_path = str(Path(self._tmp_model_dir.name) / "float_model3.onnx")
        construct_test_model1(float_model_path, activations_as_outputs=False)

        # Setup: create qdq model:
        data_reader = TestDataReader()
        qdq_model_path = str(Path(self._tmp_model_dir.name) / "qdq_model3.onnx")
        quantize_static(
            float_model_path,
            qdq_model_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=False,
            reduce_range=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )

        # Call function under test and verify all weights are present
        matched_weights = create_weight_matching(float_model_path, qdq_model_path)
        weight_names = ["W1", "W3", "W5", "B1", "B3", "B5"]
        for weight_name in weight_names:
            float_array = matched_weights[weight_name]["float"]
            dq_array = matched_weights[weight_name]["dequantized"]
            self.assertEqual(float_array.shape, dq_array.shape)

        weights_error = compute_weight_error(matched_weights)
        for weight_name in weight_names:
            self.assertIsInstance(
                weights_error[weight_name],
                float,
                f"{weight_name} quantization error {weights_error[weight_name]} too big!",
            )

    def test_create_weight_matching_per_channel(self):
        # float model
        #         (input)
        #           |
        #          Add
        #       /   |   \
        #  MatMul MatMul MatMul
        #     |     |      |
        # (output)(output)(output)
        float_model_path = str(Path(self._tmp_model_dir.name) / "float_model4.onnx")
        initializers = []
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [5, 5])
        output_tensor1 = helper.make_tensor_value_info("M", TensorProto.FLOAT, [5, 5])
        output_tensor2 = helper.make_tensor_value_info("N", TensorProto.FLOAT, [5, 5])
        output_tensor3 = helper.make_tensor_value_info("O", TensorProto.FLOAT, [5, 5])

        add_weight_data = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(add_weight_data, name="P"))
        matmul_weight_data_1 = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(matmul_weight_data_1, name="Q"))
        matmul_weight_data_2 = np.random.normal(0, 0.1, [5, 5]).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(matmul_weight_data_2, name="R"))
        initializers.append(onnx.numpy_helper.from_array(matmul_weight_data_2, name="S"))

        add_node = onnx.helper.make_node("Add", ["input", "P"], ["T"], name="Add")
        matmul_node_1 = onnx.helper.make_node("MatMul", ["T", "Q"], ["M"], name="MatMul1")
        matmul_node_2 = onnx.helper.make_node("MatMul", ["T", "R"], ["N"], name="MatMul2")
        matmul_node_3 = onnx.helper.make_node("MatMul", ["T", "S"], ["O"], name="MatMul3")

        graph = helper.make_graph(
            [add_node, matmul_node_1, matmul_node_2, matmul_node_3],
            "QDQ_Test",
            [input_tensor],
            [output_tensor1, output_tensor2, output_tensor3],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, float_model_path)

        # Setup: create qdq model:
        qdq_model_path = str(Path(self._tmp_model_dir.name) / "qdq_model4.onnx")
        quantize_static(
            float_model_path,
            qdq_model_path,
            TestDataReader([5, 5]),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            reduce_range=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )

        # Call function under test and verify all weights are present
        matched_weights = create_weight_matching(float_model_path, qdq_model_path)
        weight_names = ["P", "Q", "R", "S"]
        for weight_name in weight_names:
            float_array = matched_weights[weight_name]["float"]
            dq_array = matched_weights[weight_name]["dequantized"]
            self.assertEqual(float_array.shape, dq_array.shape)

    def test_none_test(self):
        a = np.array([2, 3, 4])
        b = np.array([7, 8, 9])
        c = np.array([1, 2, 3])
        create_activation_matching({"test" + QUANT_INPUT_SUFFIX: a, "test": c}, {"test": b})


if __name__ == "__main__":
    unittest.main()
