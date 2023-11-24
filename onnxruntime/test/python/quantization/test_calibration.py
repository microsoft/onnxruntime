#!/usr/bin/env python
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

import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader, create_calibrator


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """
    Helper function to generate initializers for test inputs
    """
    tensor = np.random.normal(0, 0.3, tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


class TestDataReader(CalibrationDataReader):
    """for test purpose"""

    def __init__(self):
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.count = 4
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


class TestCalibrateMinMaxCalibrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_calibration.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def test_augment_graph_config_1(self):
        """TEST_CONFIG_1"""

        #     Conv
        #      |
        #     Clip
        #      |
        #     MatMul

        vi_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 5, 5])
        vi_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 1, 3, 3])
        vi_e = helper.make_tensor_value_info("E", TensorProto.FLOAT, [1, 1, 5, 1])
        vi_f = helper.make_tensor_value_info("F", TensorProto.FLOAT, [1, 1, 5, 1])
        conv_node = helper.make_node(
            "Conv",
            ["A", "B"],
            ["C"],
            name="Conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        clip_node = helper.make_node("Clip", ["C"], ["D"], name="Clip")
        matmul_node = helper.make_node("MatMul", ["D", "E"], ["F"], name="MatMul")
        graph = helper.make_graph([conv_node, clip_node, matmul_node], "test_graph_1", [vi_a, vi_b, vi_e], [vi_f])

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        test_model_path = Path(self._tmp_model_dir.name).joinpath("./test_model_1.onnx")
        onnx.save(model, test_model_path.as_posix())

        # Augmenting graph
        augmented_model_path = Path(self._tmp_model_dir.name).joinpath("./augmented_test_model_1.onnx")
        calibrater = create_calibrator(test_model_path, ["Conv", "MatMul"], augmented_model_path.as_posix())
        augmented_model = calibrater.get_augment_model()

        # Checking if each added ReduceMin and ReduceMax node and its output exists
        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = [
            "A_ReduceMin",
            "A_ReduceMax",
            "B_ReduceMin",
            "B_ReduceMax",
            "C_ReduceMin",
            "C_ReduceMax",
            "D_ReduceMin",
            "D_ReduceMax",
            "E_ReduceMin",
            "E_ReduceMax",
            "F_ReduceMin",
            "F_ReduceMax",
            "A_ReduceMin_Reshape",
            "A_ReduceMax_Reshape",
            "B_ReduceMin_Reshape",
            "B_ReduceMax_Reshape",
            "C_ReduceMin_Reshape",
            "C_ReduceMax_Reshape",
            "D_ReduceMin_Reshape",
            "D_ReduceMax_Reshape",
            "E_ReduceMin_Reshape",
            "E_ReduceMax_Reshape",
            "F_ReduceMin_Reshape",
            "F_ReduceMax_Reshape",
        ]
        added_outputs = [
            "A_ReduceMin",
            "A_ReduceMax",
            "B_ReduceMin",
            "B_ReduceMax",
            "C_ReduceMin",
            "C_ReduceMax",
            "D_ReduceMin",
            "D_ReduceMax",
            "E_ReduceMin",
            "E_ReduceMax",
            "F_ReduceMin",
            "F_ReduceMax",
        ]
        # Original 3 nodes + added ReduceMin/Max nodes
        self.assertEqual(len(augmented_model_node_names), 27)
        # Original 1 graph output + added outputs * 6
        self.assertEqual(len(augmented_model_outputs), 13)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

    def test_augment_graph_config_2(self):
        """TEST_CONFIG_2"""
        #   Conv
        #    |
        #   Conv

        vi_g = helper.make_tensor_value_info("G", TensorProto.FLOAT, [1, 1, 5, 5])
        vi_h = helper.make_tensor_value_info("H", TensorProto.FLOAT, [1, 1, 3, 3])
        vi_j = helper.make_tensor_value_info("J", TensorProto.FLOAT, [1, 1, 3, 3])
        vi_k = helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, 1, 5, 5])
        conv_node_1 = helper.make_node(
            "Conv",
            ["G", "H"],
            ["I"],
            name="Conv1",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        conv_node_2 = helper.make_node(
            "Conv",
            ["I", "J"],
            ["K"],
            name="Conv2",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        graph = helper.make_graph([conv_node_1, conv_node_2], "test_graph_2", [vi_g, vi_h, vi_j], [vi_k])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        test_model_path = Path(self._tmp_model_dir.name).joinpath("./test_model_2.onnx")
        onnx.save(model, test_model_path.as_posix())

        augmented_model_path = Path(self._tmp_model_dir.name).joinpath("./augmented_test_model_2.onnx")
        calibrater = create_calibrator(test_model_path, ["Conv", "MatMul"], augmented_model_path.as_posix())
        augmented_model = calibrater.get_augment_model()

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ["I_ReduceMin", "I_ReduceMax", "K_ReduceMin", "K_ReduceMax"]
        added_outputs = ["I_ReduceMin", "I_ReduceMax", "K_ReduceMin", "K_ReduceMax"]
        # Original 2 nodes + (ReduceMin + Reshape, ReduceMax + Reshape) * 5 tensors
        self.assertEqual(len(augmented_model_node_names), 22)
        # Original 1 graph output + 5 tensors * 2
        self.assertEqual(len(augmented_model_outputs), 11)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

    def test_augment_graph_config_3(self):
        """TEST_CONFIG_3"""

        #    (input)
        #       |
        #      Relu
        #      / \
        #    Conv \
        #     |    |
        #    Clip  |
        #     |   /
        #    MatMul
        #      |
        #   (output)

        vi_l = helper.make_tensor_value_info("L", TensorProto.FLOAT, [1, 1, 5, 5])
        vi_n = helper.make_tensor_value_info("N", TensorProto.FLOAT, [1, 1, 3, 3])
        vi_q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 1, 5, 5])
        relu_node = helper.make_node("Relu", ["L"], ["M"], name="Relu")
        conv_node = helper.make_node(
            "Conv",
            ["M", "N"],
            ["O"],
            name="Conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        clip_node = helper.make_node("Clip", ["O"], ["P"], name="Clip")
        matmul_node = helper.make_node("MatMul", ["P", "M"], ["Q"], name="MatMul")
        graph = helper.make_graph([relu_node, conv_node, clip_node, matmul_node], "test_graph_3", [vi_l, vi_n], [vi_q])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        test_model_path = Path(self._tmp_model_dir.name).joinpath("./test_model_3.onnx")
        onnx.save(model, test_model_path.as_posix())

        # Augmenting graph
        augmented_model_path = Path(self._tmp_model_dir.name).joinpath("./augmented_test_model_3.onnx")
        calibrater = create_calibrator(test_model_path, ["Conv", "MatMul"], augmented_model_path.as_posix())
        augmented_model = calibrater.get_augment_model()

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = [
            "M_ReduceMin",
            "M_ReduceMax",
            "N_ReduceMin",
            "N_ReduceMax",
            "O_ReduceMin",
            "O_ReduceMax",
            "P_ReduceMin",
            "P_ReduceMax",
            "Q_ReduceMin",
            "Q_ReduceMax",
            "M_ReduceMin_Reshape",
            "M_ReduceMax_Reshape",
            "N_ReduceMin_Reshape",
            "N_ReduceMax_Reshape",
            "O_ReduceMin_Reshape",
            "O_ReduceMax_Reshape",
            "P_ReduceMin_Reshape",
            "P_ReduceMax_Reshape",
            "Q_ReduceMin_Reshape",
            "Q_ReduceMax_Reshape",
        ]
        added_outputs = [
            "M_ReduceMin",
            "M_ReduceMax",
            "N_ReduceMin",
            "N_ReduceMax",
            "O_ReduceMin",
            "O_ReduceMax",
            "P_ReduceMin",
            "P_ReduceMax",
            "Q_ReduceMin",
            "Q_ReduceMax",
        ]
        # Original 4 nodes + (ReduceMin + Reshape, ReduceMax + Reshape) * 5 tensors
        self.assertEqual(len(augmented_model_node_names), 24)
        # Original 1 graph output + 5 tensors * 2
        self.assertEqual(len(augmented_model_outputs), 11)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

    def construct_test_compute_data_model(self, test_model_path):
        #    (input)
        #       |
        #      Relu
        #      /  \
        #    Conv  \
        #     |     \
        #    Relu  Conv
        #     |     |
        #   Conv    |
        #     \     /
        #       Add
        #        |
        #       (X6)
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1, 3])
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
        graph = helper.make_graph(
            [relu_node_1, conv_node_1, relu_node_2, conv_node_2, conv_node_3, add_node],
            "test_graph_4",
            [input],
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

    def test_compute_data(self):
        test_model_path = Path(self._tmp_model_dir.name).joinpath("./test_model_4.onnx")
        self.construct_test_compute_data_model(test_model_path.as_posix())

        augmented_model_path = Path(self._tmp_model_dir.name).joinpath("./augmented_test_model_4.onnx")
        calibrater = create_calibrator(test_model_path, augmented_model_path=augmented_model_path.as_posix())
        data_reader = TestDataReader()
        calibrater.collect_data(data_reader)
        tensors_range = calibrater.compute_data()

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        infer_session = onnxruntime.InferenceSession(
            test_model_path.as_posix(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        data_reader.rewind()
        rmin = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        rmax = -1.0 * rmin
        while True:
            input = data_reader.get_next()
            if not input:
                break
            output = np.asarray(infer_session.run(None, input)).reshape(6, -1)
            rmin = np.minimum(rmin, np.amin(output, axis=1))
            rmax = np.maximum(rmax, np.amax(output, axis=1))

        min_max_pairs = list(zip(rmin, rmax))
        output_names = [infer_session.get_outputs()[i].name for i in range(len(infer_session.get_outputs()))]
        output_min_max_dict = dict(zip(output_names, min_max_pairs))
        for output_name in output_min_max_dict:
            self.assertEqual(output_min_max_dict[output_name], tensors_range[output_name].range_value)

    def test_augment_graph_with_zero_value_dimension(self):
        """TEST_CONFIG_5"""
        #   Conv
        #    |
        #   Conv
        #    |
        #  Resize

        vi_g = helper.make_tensor_value_info("G", TensorProto.FLOAT, [1, 1, 5, 5])
        vi_m = helper.make_tensor_value_info("M", TensorProto.FLOAT, [0])
        vi_n = helper.make_tensor_value_info("N", TensorProto.FLOAT, [0])
        vi_o = helper.make_tensor_value_info("O", TensorProto.FLOAT, [1, 1, 5, 5])
        # O = helper.make_tensor_value_info('O', TensorProto.FLOAT, None)
        conv_node_1 = helper.make_node(
            "Conv",
            ["G", "conv1_w"],
            ["I"],
            name="Conv1",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        conv_node_2 = helper.make_node(
            "Conv",
            ["I", "conv2_w"],
            ["K"],
            name="Conv2",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        resize_node_1 = helper.make_node("Resize", ["K", "M", "N"], ["O"], name="Reize1")
        graph = helper.make_graph(
            [conv_node_1, conv_node_2, resize_node_1],
            "test_graph_5",
            [vi_g, vi_m, vi_n],
            [vi_o],
        )
        conv1_w = generate_input_initializer([1, 1, 3, 3], np.float32, "conv1_w")
        conv2_w = generate_input_initializer([1, 1, 3, 3], np.float32, "conv2_w")
        graph.initializer.extend([conv1_w, conv2_w])

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        test_model_path = "./test_model_5.onnx"
        onnx.save(model, test_model_path)

        augmented_model_path = Path(self._tmp_model_dir.name).joinpath("./augmented_test_model_5.onnx")
        calibrater = create_calibrator(test_model_path, [], augmented_model_path.as_posix())
        augmented_model = calibrater.get_augment_model()

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = [
            "G_ReduceMin",
            "G_ReduceMax",
            "I_ReduceMin",
            "I_ReduceMax",
            "K_ReduceMin",
            "K_ReduceMax",
            "M_ReduceMin",
            "M_ReduceMax",
            "N_ReduceMin",
            "N_ReduceMax",
            "O_ReduceMin",
            "O_ReduceMax",
            "G_ReduceMin_Reshape",
            "G_ReduceMax_Reshape",
            "I_ReduceMin_Reshape",
            "I_ReduceMax_Reshape",
            "K_ReduceMin_Reshape",
            "K_ReduceMax_Reshape",
            "M_ReduceMin_Reshape",
            "M_ReduceMax_Reshape",
            "N_ReduceMin_Reshape",
            "N_ReduceMax_Reshape",
            "O_ReduceMin_Reshape",
            "O_ReduceMax_Reshape",
        ]
        added_outputs = [
            "G_ReduceMin",
            "G_ReduceMax",
            "I_ReduceMin",
            "I_ReduceMax",
            "K_ReduceMin",
            "K_ReduceMax",
            "M_ReduceMin",
            "M_ReduceMax",
            "N_ReduceMin",
            "N_ReduceMax",
            "O_ReduceMin",
            "O_ReduceMax",
        ]
        # Original 3 nodes + (ReduceMin + Reshape, ReduceMax + Reshape) * 6 tensors
        self.assertEqual(len(augmented_model_node_names), 27)
        # Original 1 graph output + 6 tensors * 2
        self.assertEqual(len(augmented_model_outputs), 13)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)


if __name__ == "__main__":
    unittest.main()
