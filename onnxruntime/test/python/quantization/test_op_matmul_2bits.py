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
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import quant_utils


class TestOpMatMul2Bits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_matmul2bits.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def fill_int2_data(self, shape: int | tuple[int, ...], symmetric: bool) -> np.ndarray:
        line = np.zeros(shape)
        line = line.reshape(-1)

        if symmetric:
            # For 2-bit symmetric: values in range [-2, 1] (excluding 0)
            v = -2.0
            for i in range(line.shape[0]):
                if v == 0:  # Skip 0 for symmetric quantization
                    v += 1
                line[i] = v
                v += 1
                if v >= 2:
                    v = -2
        else:
            # For 2-bit unsigned: values in range [0, 3]
            v = 0.0
            for i in range(line.shape[0]):
                line[i] = v
                v += 1
                if v >= 4:
                    v = 0

        return line.reshape(shape)

    def input_feeds(
        self,
        n: int,
        name2shape: dict[str, int | tuple[int, ...]],
        low: int = -1,
        high: int = 2,
        dtype: type = np.float32,
    ) -> TestDataFeeds:
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(low, high, shape).astype(dtype)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_matmul(self, output_model_path: str, symmetric: bool, k: int = 52, n: int = 288) -> None:
        #      (input)
        #         |
        #       MatMul
        #         |
        #      (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_matmul(
            input_name, weight_shape: int | tuple[int, ...], weight_name: str, output_name: str, node_name: str
        ):
            weight_data = self.fill_int2_data(weight_shape, symmetric).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))
            return onnx.helper.make_node(
                "MatMul",
                [input_name, weight_name],
                [output_name],
                node_name,
            )

        in_features = k
        out_features = n
        # make MatMul node
        matmul_node = make_matmul(
            input_name,
            [in_features, out_features],
            "linear1.weight",
            output_name,
            "MatMul_0",
        )

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [-1, in_features])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [-1, out_features])
        graph_name = "matmul_2bits_test"
        graph = helper.make_graph(
            [matmul_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        # blocked quantization requires DQ op set >= 21
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
        model.ir_version = 10  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def quant_test(
        self,
        model_fp32_path: str,
        data_reader: TestDataFeeds,
        block_size: int,
        is_symmetric: bool,
        quant_format: quant_utils.QuantFormat = quant_utils.QuantFormat.QOperator,
        op_types_to_quantize: tuple[str, ...] = ("MatMul",),
        quant_axes: tuple[tuple[str, int], ...] = (("MatMul", 0), ("Gather", 1)),
        rtol: float = 0.01,
        atol: float = 0.05,
        suffix: str = "",
    ):
        use_qdq = quant_format == quant_utils.QuantFormat.QDQ
        name_prefix = "QDQ" if use_qdq else "QOperator"
        model_int2_path = str(
            Path(self._tmp_model_dir.name)
            .joinpath(f"{name_prefix}_bs{block_size}_{is_symmetric}{suffix}.onnx")
            .absolute()
        )

        # Quantize fp32 model to int2 model
        from onnxruntime.quantization import matmul_nbits_quantizer  # noqa: PLC0415

        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))

        quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=is_symmetric,
            quant_format=quant_format,
            op_types_to_quantize=op_types_to_quantize,
            quant_axes=quant_axes,
            bits=2,
        )

        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(model, bits=2, algo_config=quant_config)
        quant.process()
        quant.model.save_model_to_file(model_int2_path, False)

        if "Gather" in op_types_to_quantize:
            quant_nodes = {"GatherBlockQuantized": 1}
        else:
            quant_nodes = {"DequantizeLinear": 1, "MatMul": 1} if use_qdq else {"MatMulNBits": 1}
        check_op_type_count(self, model_int2_path, **quant_nodes)

        if use_qdq:
            # Note: For 2-bit, we might need to use INT8/UINT8 as the actual storage type
            # since INT2/UINT2 might not be directly supported
            dq_qtype = onnx.TensorProto.INT8 if is_symmetric else onnx.TensorProto.UINT8
            dqnode_io_qtypes = (
                {
                    "DequantizeLinear": [
                        ["i", 0, dq_qtype],
                    ]
                }
                if is_symmetric
                else {
                    "DequantizeLinear": [
                        ["i", 0, dq_qtype],
                        ["i", 2, dq_qtype],
                    ]
                }
            )
            check_qtype_by_node_type(self, model_int2_path, dqnode_io_qtypes)
            for op in quant.model.opset_import():
                if op.domain in [None, "", "ai.onnx"] and op.version < 21:
                    self.fail(f"In QDQ format {op.domain} opset should be >= 21")

        data_reader.rewind()

        try:
            check_model_correctness(
                self,
                model_fp32_path,
                model_int2_path,
                data_reader.get_next(),
                rtol,
                atol,
            )
        except Exception as exception:
            if "2b quantization not yet supported on this hardware platform!" in exception.args[0]:
                # Currently we don't have int2 quantization support on all platforms, has to tolerate this exception
                pass
            else:
                raise exception

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_2bits"
    )
    def test_quantize_matmul_int2_symmetric(self):
        np.random.seed(13)

        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("matmul_fp32_symmetric.onnx").absolute())
        self.construct_model_matmul(model_fp32_path, symmetric=True)
        data_reader = self.input_feeds(1, {"input": (100, 52)})
        self.quant_test(model_fp32_path, data_reader, 32, True, rtol=0.02, atol=0.1)

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_2bits"
    )
    def test_quantize_matmul_int2_offsets(self):
        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("matmul_fp32_offset.onnx").absolute())
        self.construct_model_matmul(model_fp32_path, symmetric=False)
        data_reader = self.input_feeds(1, {"input": (100, 52)})
        self.quant_test(model_fp32_path, data_reader, 32, False, rtol=0.02, atol=0.1)


if __name__ == "__main__":
    unittest.main()
