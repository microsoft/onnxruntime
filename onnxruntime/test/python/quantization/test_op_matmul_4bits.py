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
from onnx import TensorProto, helper
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count

from onnxruntime.quantization import quant_utils


class TestOpMatMul4Bits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_matmul4bits.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def fill_int4_data(self, shape: Union[int, Tuple[int, ...]], symmetric: bool) -> np.ndarray:
        line = np.zeros(shape)
        line = line.reshape(-1)

        if symmetric:
            v = -2.0
            for i in range(line.shape[0]):
                if v == 0 or v == -3 or v == 3:
                    v += 1
                line[i] = v
                v += 1
                if v >= 8:
                    v = -8
        else:
            v = 0.0
            for i in range(line.shape[0]):
                line[i] = v
                v += 1
                if v >= 16:
                    v = 0

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

    def construct_model_matmul(self, output_model_path: str, symmetric: bool) -> None:
        #      (input)
        #         |
        #       MatMul
        #         |
        #      (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_matmul(input_name, weight_shape: Union[int, Tuple[int, ...]], weight_name: str, output_name: str):
            weight_data = self.fill_int4_data(weight_shape, symmetric).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))
            return onnx.helper.make_node(
                "MatMul",
                [input_name, weight_name],
                [output_name],
            )

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
        graph_name = "matmul_4bits_test"
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

    def quant_test(
        self,
        model_fp32_path: str,
        data_reader: TestDataFeeds,
        block_size: int,
        is_symmetric: bool,
    ):
        model_int4_path = str(
            Path(self._tmp_model_dir.name).joinpath(f"MatMulNBits_{block_size}_{is_symmetric}.onnx").absolute()
        )

        # Quantize fp32 model to int4 model
        from onnxruntime.quantization import matmul_4bits_quantizer

        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(model, block_size, is_symmetric)
        quant.process()
        quant.model.save_model_to_file(model_int4_path, False)

        quant_nodes = {"MatMulNBits": 1}
        check_op_type_count(self, model_int4_path, **quant_nodes)

        data_reader.rewind()

        try:
            check_model_correctness(self, model_fp32_path, model_int4_path, data_reader.get_next())
        except Exception as exception:
            if "4b quantization not yet supported on this hardware platform!" in exception.args[0]:
                # Currently we don't have int4 quantization support on all platforms, has to tolerate this exception
                pass
            else:
                raise exception

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_4bits"
    )
    def test_quantize_matmul_int4_symmetric(self):
        np.random.seed(13)

        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("matmul_fp32_symmetric.onnx").absolute())
        self.construct_model_matmul(model_fp32_path, symmetric=True)
        data_reader = self.input_feeds(1, {"input": [100, 52]})
        self.quant_test(model_fp32_path, data_reader, 32, True)

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_4bits"
    )
    def test_quantize_matmul_int4_offsets(self):
        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("matmul_fp32_offset.onnx").absolute())
        self.construct_model_matmul(model_fp32_path, symmetric=False)
        data_reader = self.input_feeds(1, {"input": [100, 52]})
        self.quant_test(model_fp32_path, data_reader, 32, False)


if __name__ == "__main__":
    unittest.main()
