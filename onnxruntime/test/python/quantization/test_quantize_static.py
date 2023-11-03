#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from op_test_utils import check_model_correctness, generate_random_initializer, input_feeds_neg_one_zero_one

from onnxruntime.quantization import QuantType, StaticQuantConfig, quantize, quantize_static


def construct_test_model(test_model_path, channel_size):
    """ Create an ONNX model:
    ```
        (input)
          / \
         /   \
      Conv1   \
        |      \
      Relu   Conv3
        |      |
      Conv2    |
        \\      /
           Add
            |
          output
    ```
    """

    input_value_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channel_size, 1, 3])
    output_value_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channel_size, 1, 3])

    initializer = []
    initializer.append(generate_random_initializer("W1", [channel_size, channel_size, 1, 1], np.float32))
    initializer.append(generate_random_initializer("B1", [channel_size], np.float32))
    initializer.append(generate_random_initializer("W2", [channel_size, channel_size, 1, 1], np.float32))
    initializer.append(generate_random_initializer("B2", [channel_size], np.float32))
    initializer.append(generate_random_initializer("W3", [channel_size, channel_size, 1, 1], np.float32))
    initializer.append(generate_random_initializer("B3", [channel_size], np.float32))

    nodes = []
    nodes.append(helper.make_node("Conv", ["input", "W1", "B1"], ["conv1_output"], name="conv1"))
    nodes.append(helper.make_node("Relu", ["conv1_output"], ["relu_output"], name="relu"))
    nodes.append(helper.make_node("Conv", ["relu_output", "W2", "B2"], ["conv2_output"], name="conv2"))
    nodes.append(helper.make_node("Conv", ["input", "W3", "B3"], ["conv3_output"], name="conv3"))
    nodes.append(helper.make_node("Add", ["conv2_output", "conv3_output"], ["output"], name="add"))

    graph = helper.make_graph(nodes, "test_graph_4", [input_value_info], [output_value_info], initializer=initializer)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, test_model_path)


class TestStaticQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # TODO: there will be a refactor to handle all those temporary directories.
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.quant.save.as.external")
        cls._channel_size = 16
        cls._model_fp32_path = str(Path(cls._tmp_model_dir.name) / "fp32.onnx")
        construct_test_model(cls._model_fp32_path, cls._channel_size)

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def test_save_as_external(self):
        data_reader = input_feeds_neg_one_zero_one(10, {"input": [1, self._channel_size, 1, 3]})
        for use_external_data_format in [True, False]:
            quant_model_path = str(Path(self._tmp_model_dir.name) / f"quant.{use_external_data_format}.onnx")
            quantize_static(
                self._model_fp32_path,
                quant_model_path,
                data_reader,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
                use_external_data_format=use_external_data_format,
            )

            data_reader.rewind()
            check_model_correctness(self, self._model_fp32_path, quant_model_path, data_reader.get_next())
            data_reader.rewind()

    def test_static_quant_config(self):
        data_reader = input_feeds_neg_one_zero_one(10, {"input": [1, self._channel_size, 1, 3]})
        quant_config = StaticQuantConfig(data_reader)
        quant_model_path = str(Path(self._tmp_model_dir.name) / "quant.config.onnx")
        quantize(self._model_fp32_path, quant_model_path, quant_config)

        data_reader.rewind()
        check_model_correctness(self, self._model_fp32_path, quant_model_path, data_reader.get_next())
        data_reader.rewind()

    @unittest.skip(
        "Skip failed test in Python Packaging Test Pipeline."
        "During importing neural_compressor, pycocotools throws ValueError: numpy.ndarray size changed"
    )
    def test_smooth_quant(self):
        if not find_spec("neural_compressor"):
            self.skipTest("skip test_smooth_quant since neural_compressor is not installed")
        data_reader = input_feeds_neg_one_zero_one(10, {"input": [1, self._channel_size, 1, 3]})
        quant_config = StaticQuantConfig(data_reader, extra_options={"SmoothQuant": True})
        quant_model_path = str(Path(self._tmp_model_dir.name) / "quant.config.onnx")
        quantize(self._model_fp32_path, quant_model_path, quant_config)

        data_reader.rewind()
        check_model_correctness(self, self._model_fp32_path, quant_model_path, data_reader.get_next())
        data_reader.rewind()

        model = onnx.load(quant_model_path)
        self.assertIn("Mul", [i.op_type for i in model.graph.node])


if __name__ == "__main__":
    unittest.main()
