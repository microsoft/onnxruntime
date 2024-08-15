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
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

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

    def input_feeds(
        self,
        n: int,
        name2shape: Dict[str, Union[int, Tuple[int, ...]]],
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

    def construct_model_matmul(self, output_model_path: str, symmetric: bool) -> None:
        #      (input)
        #         |
        #       MatMul
        #         |
        #      (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_matmul(
            input_name, weight_shape: Union[int, Tuple[int, ...]], weight_name: str, output_name: str, node_name: str
        ):
            weight_data = self.fill_int4_data(weight_shape, symmetric).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))
            return onnx.helper.make_node(
                "MatMul",
                [input_name, weight_name],
                [output_name],
                node_name,
            )

        in_features = 52
        out_features = 288
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
        graph_name = "matmul_4bits_test"
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

    def construct_model_gather(
        self,
        output_model_path: str,
        symmetric: bool,
        tdata: TensorProto.DataType,
        tind: TensorProto.DataType,
        vocab_size: int = 545,
        embedding_len: int = 228,
    ) -> None:
        #      (input)
        #         |
        #       Gather
        #         |
        #      (output)
        indices_name = "input"
        output_name = "output"
        initializers = []

        def make_gather(
            indices_name, data_shape: Union[int, Tuple[int, ...]], data_name: str, output_name: str, node_name: str
        ):
            weight_data = self.fill_int4_data(data_shape, symmetric).astype(
                np.float32 if tdata == TensorProto.FLOAT else np.float16
            )
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=data_name))
            kwargs = {"axis": 0}
            return onnx.helper.make_node(
                "Gather",
                [data_name, indices_name],
                [output_name],
                node_name,
                **kwargs,
            )

        gather_node = make_gather(
            indices_name,
            (vocab_size, embedding_len),
            "linear1.weight",
            output_name,
            "Gather_0",
        )

        # make graph
        input_tensor = helper.make_tensor_value_info(indices_name, tind, [-1, 1000])
        output_tensor = helper.make_tensor_value_info(output_name, tdata, [-1, 1000, embedding_len])
        graph_name = "gather_4bits_test"
        graph = helper.make_graph(
            [gather_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        # QDQ and gather requires op set >= 21. The tool should automatically update the opset.
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
        model.ir_version = 9  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def quant_test(
        self,
        model_fp32_path: str,
        data_reader: TestDataFeeds,
        block_size: int,
        is_symmetric: bool,
        quant_format: quant_utils.QuantFormat = quant_utils.QuantFormat.QOperator,
        op_types_to_quantize: Tuple[str, ...] = ("MatMul",),
        quant_axes: Tuple[Tuple[str, int], ...] = (("MatMul", 0), ("Gather", 1)),
        rtol: float = 0.01,
        atol: float = 0.05,
    ):
        use_qdq = quant_format == quant_utils.QuantFormat.QDQ
        name_prefix = "QDQ" if use_qdq else "QOperator"
        model_int4_path = str(
            Path(self._tmp_model_dir.name).joinpath(f"{name_prefix}_{block_size}_{is_symmetric}.onnx").absolute()
        )

        # Quantize fp32 model to int4 model
        from onnxruntime.quantization import matmul_4bits_quantizer

        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
        quant_config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=is_symmetric,
            quant_format=quant_format,
            op_types_to_quantize=op_types_to_quantize,
            quant_axes=quant_axes,
        )
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(model, algo_config=quant_config)
        quant.process()
        quant.model.save_model_to_file(model_int4_path, False)

        if "Gather" in op_types_to_quantize:
            quant_nodes = {"GatherBlockQuantized": 1}
        else:
            quant_nodes = {"DequantizeLinear": 1, "MatMul": 1} if use_qdq else {"MatMulNBits": 1}
        check_op_type_count(self, model_int4_path, **quant_nodes)

        if use_qdq:
            dq_qtype = TensorProto.INT4 if is_symmetric else TensorProto.UINT4
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
            check_qtype_by_node_type(self, model_int4_path, dqnode_io_qtypes)
            for op in quant.model.opset_import():
                if op.domain in [None, "", "ai.onnx"] and op.version < 21:
                    self.fail(f"In QDQ format {op.domain} opset should be >= 21")

        data_reader.rewind()

        try:
            check_model_correctness(self, model_fp32_path, model_int4_path, data_reader.get_next(), rtol, atol)
        except Exception as exception:
            if "4b quantization not yet supported on this hardware platform!" in exception.args[0]:
                # Currently we don't have int4 quantization support on all platforms, has to tolerate this exception
                pass
            else:
                raise exception

    def quant_test_with_algo(
        self,
        algorithm: str,
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

        algo_config = None
        if algorithm == "RTN":
            # test RTN algorithm
            algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig()
        elif algorithm == "GPTQ":
            # test GPTQ algorithm
            algo_config = matmul_4bits_quantizer.GPTQWeightOnlyQuantConfig(calibration_data_reader=data_reader)
        elif algorithm == "HQQ":
            # test HQQ algorithm
            algo_config = matmul_4bits_quantizer.HQQWeightOnlyQuantConfig(block_size=block_size)

        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(model, block_size, is_symmetric, algo_config=algo_config)
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
        data_reader = self.input_feeds(1, {"input": (100, 52)})
        self.quant_test(model_fp32_path, data_reader, 32, True)

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_4bits"
    )
    def test_quantize_matmul_int4_offsets(self):
        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("matmul_fp32_offset.onnx").absolute())
        self.construct_model_matmul(model_fp32_path, symmetric=False)
        data_reader = self.input_feeds(1, {"input": (100, 52)})
        self.quant_test(model_fp32_path, data_reader, 32, False)

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_4bits"
    )
    def test_quantize_gather_int4_symmetric(self):
        np.random.seed(13)

        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("gather_fp32_symmetric.onnx").absolute())
        self.construct_model_gather(model_fp32_path, True, TensorProto.FLOAT, TensorProto.INT32)
        data_reader = self.input_feeds(1, {"input": (100, 1000)}, -545, 535, np.int32)
        # cover rounding error
        self.quant_test(model_fp32_path, data_reader, 32, True, op_types_to_quantize=("Gather",), rtol=0.2, atol=0.5)

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_4bits"
    )
    def test_quantize_gather_int4_offsets(self):
        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("gather_fp32_offset.onnx").absolute())
        self.construct_model_gather(model_fp32_path, False, TensorProto.FLOAT16, TensorProto.INT64)
        data_reader = self.input_feeds(1, {"input": (100, 1000)}, -545, 535, np.int64)
        # cover rounding error
        self.quant_test(model_fp32_path, data_reader, 32, False, op_types_to_quantize=("Gather",), rtol=0.2, atol=0.5)

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_4bits"
    )
    def test_quantize_matmul_int4_symmetric_qdq(self):
        np.random.seed(13)

        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("matmul_fp32_symmetric.onnx").absolute())
        self.construct_model_matmul(model_fp32_path, symmetric=True)
        data_reader = self.input_feeds(1, {"input": (100, 52)})
        self.quant_test(model_fp32_path, data_reader, 32, True, quant_utils.QuantFormat.QDQ)

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_4bits"
    )
    def test_quantize_matmul_int4_offsets_qdq(self):
        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("matmul_fp32_offset.onnx").absolute())
        self.construct_model_matmul(model_fp32_path, symmetric=False)
        data_reader = self.input_feeds(1, {"input": (100, 52)})
        self.quant_test(model_fp32_path, data_reader, 32, False, quant_utils.QuantFormat.QDQ)

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_4bits"
    )
    def test_quantize_matmul_int4_using_rtn_algo(self):
        if not find_spec("neural_compressor"):
            self.skipTest("skip test_smooth_quant since neural_compressor is not installed")
        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("matmul_fp32_offset.onnx").absolute())
        self.construct_model_matmul(model_fp32_path, symmetric=False)
        data_reader = self.input_feeds(1, {"input": (100, 52)})
        self.quant_test_with_algo("RTN", model_fp32_path, data_reader, 32, False)

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_4bits"
    )
    def test_quantize_matmul_int4_using_gptq_algo(self):
        if not find_spec("neural_compressor"):
            self.skipTest("skip test_smooth_quant since neural_compressor is not installed")
        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("matmul_fp32_offset.onnx").absolute())
        self.construct_model_matmul(model_fp32_path, symmetric=False)
        data_reader = self.input_feeds(1, {"input": (100, 52)})
        self.quant_test_with_algo("GPTQ", model_fp32_path, data_reader, 32, False)

    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_4bits"
    )
    def test_quantize_matmul_int4_using_hqq_algo(self):
        if not find_spec("torch"):
            self.skipTest("skip test_hqq_quant since torch is not installed")
        model_fp32_path = str(Path(self._tmp_model_dir.name).joinpath("matmul_fp32_offset.onnx").absolute())
        self.construct_model_matmul(model_fp32_path, symmetric=False)
        data_reader = self.input_feeds(1, {"input": (100, 52)})
        self.quant_test_with_algo("HQQ", model_fp32_path, data_reader, 32, False)


if __name__ == "__main__":
    unittest.main()
