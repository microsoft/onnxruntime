# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Tests for the save_activations module."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.save_activations import collect_activations, modify_model_output_intermediate_tensors


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """
    Helper function to generate initializers for test inputs
    """
    tensor = np.random.normal(0, 0.3, tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


def construct_test_model1(test_model_path):
    """ Create an ONNX model shaped as:
    ```
       (input)
          |
         Relu
         /  \
       Conv  \
        |     \
       Relu  Conv
        |     |
      Conv    |
        \     /
          Add
           |
          (X6)
    ```
    We are keeping all intermediate tensors as output, just for test verification
    purposes
    """

    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1, 3])
    x1_output = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 3, 1, 3])
    x2_output = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 3, 1, 3])
    x3_output = helper.make_tensor_value_info("X3", TensorProto.FLOAT, [1, 3, 1, 3])
    x4_output = helper.make_tensor_value_info("X4", TensorProto.FLOAT, [1, 3, 1, 3])
    x5_output = helper.make_tensor_value_info("X5", TensorProto.FLOAT, [1, 3, 1, 3])
    x6_output = helper.make_tensor_value_info("X6", TensorProto.FLOAT, [1, 3, 1, 3])
    w1 = generate_input_initializer([3, 3, 1, 1], np.float32, "W1")
    b1 = generate_input_initializer([3], np.float32, "B1")
    w3 = generate_input_initializer([3, 3, 1, 1], np.float32, "W3")
    b3 = generate_input_initializer([3], np.float32, "B3")
    w5 = generate_input_initializer([3, 3, 1, 1], np.float32, "W5")
    b5 = generate_input_initializer([3], np.float32, "B5")
    relu_node_1 = helper.make_node("Relu", ["input"], ["X1"], name="Relu1")
    conv_node_1 = helper.make_node("Conv", ["X1", "W1", "B1"], ["X2"], name="Conv1")
    relu_node_2 = helper.make_node("Relu", ["X2"], ["X3"], name="Relu2")
    conv_node_2 = helper.make_node("Conv", ["X3", "W3", "B3"], ["X4"], name="Conv2")
    conv_node_3 = helper.make_node("Conv", ["X1", "W5", "B5"], ["X5"], name="Conv3")
    add_node = helper.make_node("Add", ["X4", "X5"], ["X6"], name="Add")

    # we are keeping all tensors in the output anyway for verification purpose
    graph = helper.make_graph(
        [relu_node_1, conv_node_1, relu_node_2, conv_node_2, conv_node_3, add_node],
        "test_graph_4",
        [input_vi],
        [x1_output, x2_output, x3_output, x4_output, x5_output, x6_output],
    )
    graph.initializer.add().CopyFrom(w1)
    graph.initializer.add().CopyFrom(b1)
    graph.initializer.add().CopyFrom(w3)
    graph.initializer.add().CopyFrom(b3)
    graph.initializer.add().CopyFrom(w5)
    graph.initializer.add().CopyFrom(b5)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, test_model_path)


class TestDataReader(CalibrationDataReader):
    """Random Data Input Generator"""

    def __init__(self):
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.count = 2
        self.input_data_list = []
        for _ in range(self.count):
            self.input_data_list.append(np.random.normal(0, 0.33, [1, 3, 1, 3]).astype(np.float32))

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            input_name = "input"
            self.enum_data_dicts = iter([{input_name: input_data} for input_data in self.input_data_list])
        return next(self.enum_data_dicts, None)

    def rewind(self):
        self.preprocess_flag = True


class TestSaveActivations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_save_activations.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def test_saved_tensors_match_internal_tensors(self):
        test_model_path = str(Path(self._tmp_model_dir.name) / "augmented_model.onnx")
        construct_test_model1(test_model_path)
        data_reader = TestDataReader()

        aug_model = modify_model_output_intermediate_tensors(test_model_path)
        augmented_model_path = str(Path(self._tmp_model_dir.name).joinpath("augmented_test_model_1.onnx"))

        onnx.save(
            aug_model,
            augmented_model_path,
            save_as_external_data=False,
        )

        tensor_dict = collect_activations(augmented_model_path, data_reader)

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


if __name__ == "__main__":
    unittest.main()
