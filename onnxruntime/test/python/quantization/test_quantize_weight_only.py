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
from onnxruntime.quantization.onnx_model import ONNXModel
from op_test_utils import check_model_correctness, input_feeds_neg_one_zero_one

from onnxruntime.quantization import (
    RTNWeightOnlyQuantConfig,
    GPTQWeightOnlyQuantConfig, 
    quantize_weight_only
)

def construct_model(output_model_path):
    #      (input)
    #         |
    #        Mul
    #         |
    #       MatMul
    #         |
    #      (output)
    initializers = []
    
    # make mul node
    mul_data = np.random.normal(0, 0.1, [1, 10]).astype(np.float32)
    initializers.append(onnx.numpy_helper.from_array(mul_data, name="mul.data"))
    mul_node = onnx.helper.make_node("Mul", ["input", "mul.data"], ["mul.output"], "Mul_0")

    # make matmul node
    matmul_weight = np.random.normal(0, 0.1, [10, 1]).astype(np.float32)
    initializers.append(onnx.numpy_helper.from_array(matmul_weight, name="matmul.weight"))
    matmul_node = onnx.helper.make_node("MatMul", 
                                        ["mul.output", "matmul.weight"], 
                                        ["output"],
                                        "MatMul_1")

    # make graph
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    graph_name = "weight_only_quant_test"
    graph = helper.make_graph(
        [mul_node, matmul_node],
        graph_name,
        [input_tensor],
        [output_tensor],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = onnx.IR_VERSION

    onnx.save(model, output_model_path)

class TestWeightOnlyQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # TODO: there will be a refactor to handle all those temporary directories.
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="ort.quant.save.as.external")
        cls._model_fp32_path = str(Path(cls._tmp_model_dir.name) / "fp32.onnx")
        cls._model_weight_only_path = str(Path(cls._tmp_model_dir.name) / "fp32.weight_only_quant.onnx")
        np.random.seed(1)
        construct_model(cls._model_fp32_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    @unittest.skip(
        "Skip failed test in Python Packaging Test Pipeline."
        "During importing neural_compressor, pycocotools throws ValueError: numpy.ndarray size changed"
    )
    def test_quantize_weight_only_rtn(self):
        if not find_spec("neural_compressor"):
            self.skipTest("skip test_quantize_weight_only_rtn since neural_compressor is not installed")
            
        weight_only_config = RTNWeightOnlyQuantConfig()
        quantize_weight_only(self._model_fp32_path, self._model_weight_only_path, weight_only_config)
        check_model_correctness(
            self,
            self._model_fp32_path,
            self._model_weight_only_path,
            {"input": np.random.rand(1, 10).astype(np.float32)},
        )

        model_fp32 = ONNXModel(onnx.load(self._model_fp32_path))
        model_weight_only = ONNXModel(onnx.load(self._model_weight_only_path))
        self.assertNotEqual(model_fp32.get_initializer("matmul.weight"), 
                            model_weight_only.get_initializer("matmul.weight"))


    @unittest.skip(
        "Skip failed test in Python Packaging Test Pipeline."
        "During importing neural_compressor, pycocotools throws ValueError: numpy.ndarray size changed"
    )
    def test_quantize_weight_only_gptq(self):
        if not find_spec("neural_compressor"):
            self.skipTest("skip test_quantize_weight_only_gptq since neural_compressor is not installed")
        
        data_reader = input_feeds_neg_one_zero_one(10, {"input": [1, 10]})
        weight_only_config = GPTQWeightOnlyQuantConfig(data_reader)
        quantize_weight_only(self._model_fp32_path, self._model_weight_only_path, weight_only_config)
        check_model_correctness(
            self,
            self._model_fp32_path,
            self._model_weight_only_path,
            {"input": np.random.rand(1, 10).astype(np.float32)},
        )

        model_fp32 = ONNXModel(onnx.load(self._model_fp32_path))
        model_weight_only = ONNXModel(onnx.load(self._model_weight_only_path))
        self.assertNotEqual(model_fp32.get_initializer("matmul.weight"), 
                            model_weight_only.get_initializer("matmul.weight"))

if __name__ == '__main__':
    unittest.main()
