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
from typing import Dict, Tuple, Union

import numpy as np
import onnx
import parameterized
from onnx import TensorProto, helper
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count

from onnxruntime.quantization import quant_utils

quant_maps = {
    0: [
        0.00000000,
        5.208333333e-03,
        0.66666667,
        1.00000000,
        0.33333333,
        0.50000000,
        0.16666667,
        0.25000000,
        -0.00000000,
        -5.208333333e-03,
        -0.66666667,
        -1.00000000,
        -0.33333333,
        -0.50000000,
        -0.16666667,
        -0.25000000,
    ],
    1: [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
}


class TestOpMatMulBnb4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_matmulbnb4.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def fill_bnb4_data(self, shape: Tuple[int, int], quant_type: int) -> np.ndarray:
        rows, cols = shape
        line = np.zeros(shape)
        line = line.reshape(-1)
        quant_map = np.array(quant_maps[quant_type], dtype=np.float32)

        v = 0
        for i in range(line.shape[0]):
            line[i] = quant_map[v]
            v += 1
            if v >= 16:
                v = 0

        # bnb quantization quantizes weight.T after flattening
        line = line.reshape(cols, rows).transpose()
        return line.reshape(shape)

    def input_feeds(self, n: int, name2shape: Dict[str, Union[int, Tuple[int, ...]]]) -> TestDataFeeds:
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_matmul(self, output_model_path: str, quant_type: int) -> None:
        #      (input)
        #         |
        #       MatMul
        #         |
        #      (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_matmul(input_name, weight_shape: Union[int, Tuple[int, ...]], weight_name: str, output_name: str):
            weight_data = self.fill_bnb4_data(weight_shape, quant_type).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))
            return onnx.helper.make_node(
                "MatMul",
                [input_name, weight_name],
                [output_name],
            )

        # for this to work (in_features * out_features) % block_size == 0
        in_features = 52
        out_features = 288
        # make MatMul node
        matmul_node = make_matmul(
            input_name,
            [in_features, out_features],
            "linear1.weight",
            output_name,
        )

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [-1, in_features])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [-1, out_features])
        graph_name = "matmul_bnb4_test"
        graph = helper.make_graph(
            [matmul_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def quant_test(self, model_fp32_path: str, data_reader: TestDataFeeds, quant_type: int, block_size: int):
        model_bnb4_path = str(
            Path(self._tmp_model_dir.name).joinpath(f"MatMulBnb4_{quant_type}_{block_size}.onnx").absolute()
        )

        # Quantize fp32 model to bnb4 model
        from onnxruntime.quantization import matmul_bnb4_quantizer

        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
        quant = matmul_bnb4_quantizer.MatMulBnb4Quantizer(model, quant_type, block_size)
        quant.process()
        quant.model.save_model_to_file(model_bnb4_path, False)

        quant_nodes = {"MatMulBnb4": 1}
        check_op_type_count(self, model_bnb4_path, **quant_nodes)

        data_reader.rewind()

        try:
            check_model_correctness(self, model_fp32_path, model_bnb4_path, data_reader.get_next())
        except Exception as exception:
            raise exception

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_bnb4"
    )
    @parameterized.parameterized.expand([0, 1])
    def test_quantize_matmul_bnb4(self, quant_type):
        np.random.seed(13)

        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath(f"matmul_fp32_{quant_type}.onnx").absolute())
        self.construct_model_matmul(model_fp32_path, quant_type)
        data_reader = self.input_feeds(1, {"input": [100, 52]})
        self.quant_test(model_fp32_path, data_reader, quant_type, 64)


if __name__ == "__main__":
    unittest.main()
