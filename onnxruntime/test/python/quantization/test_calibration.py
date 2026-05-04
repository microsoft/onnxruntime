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
from onnxruntime.quantization import quantize_static
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    CalibrationMethod,
    TensorData,
    TensorsData,
    create_calibrator,
    load_tensors_data,
    save_tensors_data,
)


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

    def construct_test_compute_data_model(self, test_model_path, opset_version=13, augmented=True):
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
        graph_outputs = None
        if augmented:
            graph_outputs = [
                helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 3, 1, 3]),
                helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 3, 1, 3]),
                helper.make_tensor_value_info("X3", TensorProto.FLOAT, [1, 3, 1, 3]),
                helper.make_tensor_value_info("X4", TensorProto.FLOAT, [1, 3, 1, 3]),
                helper.make_tensor_value_info("X5", TensorProto.FLOAT, [1, 3, 1, 3]),
                helper.make_tensor_value_info("X6", TensorProto.FLOAT, [1, 3, 1, 3]),
            ]
        else:
            graph_outputs = [helper.make_tensor_value_info("X6", TensorProto.FLOAT, [1, 3, 1, 3])]

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
            graph_outputs,
        )
        graph.initializer.add().CopyFrom(w1)
        graph.initializer.add().CopyFrom(b1)
        graph.initializer.add().CopyFrom(w3)
        graph.initializer.add().CopyFrom(b3)
        graph.initializer.add().CopyFrom(w5)
        graph.initializer.add().CopyFrom(b5)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
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

        min_max_pairs = list(zip(rmin, rmax, strict=False))
        output_names = [infer_session.get_outputs()[i].name for i in range(len(infer_session.get_outputs()))]
        output_min_max_dict = dict(zip(output_names, min_max_pairs, strict=False))
        for output_name, min_max in output_min_max_dict.items():
            self.assertEqual(min_max, tensors_range[output_name].range_value)

    def test_histogram_calibrators_run(self):
        """
        Runs all histogram-based calibrators (Percentile, Entropy, Distribution) and checks that they run
        and generate the expected number of tensor ranges. Does not check correctness of range values.
        """
        # Create test model.
        test_model_path = Path(self._tmp_model_dir.name).joinpath("./test_model_4.onnx")
        self.construct_test_compute_data_model(test_model_path.as_posix(), augmented=False)

        # Count the number of tensors in the model.
        model = onnx.load_model(test_model_path)
        model = onnx.shape_inference.infer_shapes(model)
        num_tensors = len(model.graph.value_info) + len(model.graph.input) + len(model.graph.output)

        # Run all histogram calibration methods.
        data_reader = TestDataReader()
        calibration_methods = [CalibrationMethod.Percentile, CalibrationMethod.Entropy, CalibrationMethod.Distribution]
        for calibration_method in calibration_methods:
            with self.subTest(calibration_method=calibration_method):
                data_reader.rewind()
                augmented_model_path = Path(self._tmp_model_dir.name).joinpath(f"augmented_{calibration_method}.onnx")
                calibrator = create_calibrator(
                    test_model_path, calibrate_method=calibration_method, augmented_model_path=augmented_model_path
                )
                calibrator.collect_data(data_reader)
                tensors_range = calibrator.compute_data()
                self.assertEqual(len(tensors_range.items()), num_tensors)  # A range for every tensor in the graph.

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

    def test_compute_data_per_channel(self):
        test_model_path = Path(self._tmp_model_dir.name).joinpath("./test_model_6.onnx")
        self.construct_test_compute_data_model(test_model_path.as_posix(), opset_version=18)

        augmented_model_path = Path(self._tmp_model_dir.name).joinpath("./augmented_test_model_6.onnx")
        calibrater = create_calibrator(
            test_model_path, augmented_model_path=augmented_model_path.as_posix(), extra_options={"per_channel": True}
        )
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
        rmin = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)[:, np.newaxis]
        rmax = -1.0 * rmin
        while True:
            input = data_reader.get_next()
            if not input:
                break
            output = np.asarray(infer_session.run(None, input)).reshape((6, 3, -1))
            rmin = np.minimum(rmin, np.amin(output, axis=-1))
            rmax = np.maximum(rmax, np.amax(output, axis=-1))

        min_max_pairs = list(zip(rmin, rmax, strict=False))
        output_names = [infer_session.get_outputs()[i].name for i in range(len(infer_session.get_outputs()))]
        output_min_max_dict = dict(zip(output_names, min_max_pairs, strict=False))
        for output_name, min_max in output_min_max_dict.items():
            np.testing.assert_equal(min_max, tensors_range[output_name].range_value)


class TestCalibrationCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory(prefix="test_calibration_cache.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()

    def _make_simple_model(self, path):
        """Build a tiny Conv+Relu model for end-to-end cache tests."""
        vi_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1, 3])
        vi_output = helper.make_tensor_value_info("X6", TensorProto.FLOAT, [1, 3, 1, 3])
        w1 = generate_input_initializer([3, 3, 1, 1], np.float32, "W1")
        b1 = generate_input_initializer([3], np.float32, "B1")
        conv_node = helper.make_node("Conv", ["input", "W1", "B1"], ["X2"], name="Conv1")
        relu_node = helper.make_node("Relu", ["X2"], ["X6"], name="Relu1")
        graph = helper.make_graph([conv_node, relu_node], "cache_test_graph", [vi_input], [vi_output])
        graph.initializer.add().CopyFrom(w1)
        graph.initializer.add().CopyFrom(b1)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, path)

    def test_save_load_tensors_data_minmax_roundtrip(self):
        td = TensorsData(
            CalibrationMethod.MinMax,
            {"x": TensorData(lowest=np.array(-1.0, dtype=np.float32), highest=np.array(2.0, dtype=np.float32))},
        )
        cache_path = Path(self._tmp_dir.name) / "minmax_cache.json"
        save_tensors_data(td, cache_path)
        self.assertTrue(cache_path.exists())

        loaded = load_tensors_data(cache_path)
        self.assertEqual(loaded.calibration_method, CalibrationMethod.MinMax)
        self.assertEqual(list(loaded.keys()), ["x"])
        lo, hi = loaded["x"].range_value
        np.testing.assert_array_equal(lo, np.array(-1.0, dtype=np.float32))
        np.testing.assert_array_equal(hi, np.array(2.0, dtype=np.float32))
        self.assertEqual(lo.shape, ())
        self.assertEqual(hi.shape, ())

    def test_save_load_tensors_data_entropy_roundtrip(self):
        hist = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        hist_edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        td = TensorsData(
            CalibrationMethod.Entropy,
            {
                "y": TensorData(
                    lowest=np.array(-0.5, dtype=np.float32),
                    highest=np.array(0.5, dtype=np.float32),
                    hist=hist,
                    hist_edges=hist_edges,
                )
            },
        )
        cache_path = Path(self._tmp_dir.name) / "entropy_cache.json"
        save_tensors_data(td, cache_path)

        loaded = load_tensors_data(cache_path)
        self.assertEqual(loaded.calibration_method, CalibrationMethod.Entropy)
        lo, hi = loaded["y"].range_value
        np.testing.assert_array_almost_equal(lo, np.array(-0.5, dtype=np.float32))
        np.testing.assert_array_almost_equal(hi, np.array(0.5, dtype=np.float32))
        np.testing.assert_array_almost_equal(loaded["y"].hist, hist)
        np.testing.assert_array_almost_equal(loaded["y"].hist_edges, hist_edges)

    def test_load_tensors_data_invalid_path(self):
        bogus = Path(self._tmp_dir.name) / "does_not_exist.json"
        with self.assertRaises(FileNotFoundError):
            load_tensors_data(bogus)

    def test_quantize_static_calibration_cache_path(self):
        model_path = Path(self._tmp_dir.name) / "tiny_model.onnx"
        self._make_simple_model(str(model_path))

        cache_path = Path(self._tmp_dir.name) / "quant_cache.json"
        out1_path = Path(self._tmp_dir.name) / "quantized1.onnx"
        out2_path = Path(self._tmp_dir.name) / "quantized2.onnx"

        # First call: calibration_data_reader provided, cache written
        data_reader = TestDataReader()
        quantize_static(
            str(model_path),
            str(out1_path),
            calibration_data_reader=data_reader,
            calibration_cache_path=cache_path,
        )
        self.assertTrue(cache_path.exists())
        td1 = load_tensors_data(cache_path)

        # Second call: no data_reader, load from cache
        quantize_static(
            str(model_path),
            str(out2_path),
            calibration_data_reader=None,
            calibration_cache_path=cache_path,
        )
        self.assertTrue(out2_path.exists())
        td2 = load_tensors_data(cache_path)
        self.assertEqual(td1.calibration_method, td2.calibration_method)

    def test_quantize_static_no_reader_no_cache_raises(self):
        model_path = Path(self._tmp_dir.name) / "tiny_model2.onnx"
        self._make_simple_model(str(model_path))
        out_path = Path(self._tmp_dir.name) / "quantized_err.onnx"

        with self.assertRaises(ValueError):
            quantize_static(str(model_path), str(out_path), calibration_data_reader=None)

    def test_save_tensors_data_creates_parent_dir(self):
        nested_path = Path(self._tmp_dir.name) / "nested" / "dir" / "cache.json"
        td = TensorsData(
            CalibrationMethod.MinMax,
            {"x": TensorData(lowest=np.array(-1.0, dtype=np.float32), highest=np.array(1.0, dtype=np.float32))},
        )
        save_tensors_data(td, nested_path)
        self.assertTrue(nested_path.exists())

    def test_save_tensors_data_handles_scalar_bins(self):
        td = TensorsData(
            CalibrationMethod.Entropy,
            {
                "z": TensorData(
                    lowest=np.array(0.0, dtype=np.float32),
                    highest=np.array(1.0, dtype=np.float32),
                    hist=np.array([1, 2], dtype=np.int64),
                    bins=np.int64(5),
                )
            },
        )
        cache_path = Path(self._tmp_dir.name) / "scalar_bins_cache.json"
        save_tensors_data(td, cache_path)
        loaded = load_tensors_data(cache_path)
        self.assertEqual(loaded["z"].bins, 5)

    def test_load_tensors_data_method_mismatch_raises(self):
        model_path = Path(self._tmp_dir.name) / "tiny_mismatch.onnx"
        self._make_simple_model(str(model_path))
        cache_path = Path(self._tmp_dir.name) / "mismatch_cache.json"
        out_path = Path(self._tmp_dir.name) / "quantized_mismatch.onnx"

        data_reader = TestDataReader()
        quantize_static(
            str(model_path),
            str(out_path),
            calibration_data_reader=data_reader,
            calibrate_method=CalibrationMethod.MinMax,
            calibration_cache_path=cache_path,
        )

        with self.assertRaises(ValueError):
            quantize_static(
                str(model_path),
                str(out_path),
                calibration_data_reader=None,
                calibrate_method=CalibrationMethod.Entropy,
                calibration_cache_path=cache_path,
            )


if __name__ == "__main__":
    unittest.main()
