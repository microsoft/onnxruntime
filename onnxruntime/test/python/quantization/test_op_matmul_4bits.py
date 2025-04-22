#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import tempfile
import unittest
from importlib.util import find_spec
from itertools import product
from parameterized import parameterized
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import quant_utils, matmul_4bits_quantizer


class TestOpMatMul4Bits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_matmul4bits.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def fill_weight_data(self, llama_cpp_quant_type_name: str, shape: int | tuple[int, ...]) -> np.ndarray:
        line = np.zeros(shape)

        # type_0: w = d_fp16 * q
        # type_1: w = d_fp16 * q + m_fp16
        # q2_k: w = d_fp16 * q + m_fp16
        # q6_k: w = d_fp16[i] * scales_q8[j] * q
        # q8_k: *y++ = x[i].d * x[i].qs[j];
        # following is not precisely correct, but it is ok for test purpose
        # it may be better to quand and dequant the weight
        # to take the idempotent nature of the quantization
        if llama_cpp_quant_type_name == "q4_0":
            symmetric = True
            start_value = -2.0
            max_value = 7
            min_value = -8
        elif llama_cpp_quant_type_name == "q4_1":
            symmetric = False
            start_value = 0.0
            max_value = 15
            min_value = 0
        elif llama_cpp_quant_type_name == "q5_0":
            symmetric = True
            start_value =-2.0
            max_value = 15
            min_value = -16
        elif llama_cpp_quant_type_name == "q5_1":
            symmetric = False
            start_value =0.0
            max_value = 31
            min_value = 0
        elif llama_cpp_quant_type_name == "q8_0":
            symmetric = True
            start_value = -2.0
            max_value = -128
            min_value = 127
        elif llama_cpp_quant_type_name == "q2_K":
            symmetric = False
            start_value = -1.0
            max_value = 1
            min_value = -2
        elif llama_cpp_quant_type_name == "q3_K":
            symmetric = False
            start_value = -3.0
            max_value = 3
            min_value = -4
        elif llama_cpp_quant_type_name == "q4_K":
            symmetric = False
            start_value = -2.0
            max_value = 7
            min_value = -8
        elif llama_cpp_quant_type_name == "q5_K":
            symmetric = False
            start_value = 0.0
            max_value = 31
            min_value = 0
        elif llama_cpp_quant_type_name == "q6_K":
            symmetric = False
            start_value = 0
            max_value = 63
            min_value = 0
        elif llama_cpp_quant_type_name in ["tq1_0", "tq2_0"]:
            symmetric = False
            start_value = -1.0
            max_value = 1
            min_value = -1
        else:
            raise ValueError(f"Unsupported llama_cpp_quant_type_name: {llama_cpp_quant_type_name}")

        v = start_value
        for c in range(line.shape[1]):
            for r in range(line.shape[0]):
                if symmetric:
                    if v == 0 or v == -3 or v == 3:
                        v += 1
                    line[r][c] = v
                    v += 1
                    if v > max_value:
                        v = min_value
                else:
                    line[r][c] = v
                    v += 1
                    if v > max_value:
                        v = min_value

        return line

    def fill_int4_data(self, shape: int | tuple[int, ...], symmetric: bool) -> np.ndarray:
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

    def construct_model_matmul(self, output_model_path: str, symmetric: bool,
                            in_features: int = 52, out_features: int = 288,
                            llama_cpp_quant_type_name: str="") -> None:
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_matmul(
            input_name, weight_shape: int | tuple[int, ...], weight_name: str, output_name: str, node_name: str
        ):
            if llama_cpp_quant_type_name:
                weight_data = self.fill_weight_data(llama_cpp_quant_type_name, weight_shape).astype(np.float32)
            else:
                weight_data = self.fill_int4_data(weight_shape, symmetric).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))
            return onnx.helper.make_node(
                "MatMul",
                [input_name, weight_name],
                [output_name],
                node_name,
            )

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
            indices_name, data_shape: int | tuple[int, ...], data_name: str, output_name: str, node_name: str
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
        op_types_to_quantize: tuple[str, ...] = ("MatMul",),
        quant_axes: tuple[tuple[str, int], ...] = (("MatMul", 0), ("Gather", 1)),
        rtol: float = 0.01,
        atol: float = 0.05,
    ):
        use_qdq = quant_format == quant_utils.QuantFormat.QDQ
        name_prefix = "QDQ" if use_qdq else "QOperator"
        model_int4_path = str(
            Path(self._tmp_model_dir.name).joinpath(f"{name_prefix}_{block_size}_{is_symmetric}.onnx").absolute()
        )

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


    def test_quantize_llama_cpp_phi_35_mini_4k_instruct(self):
        model_fp32_path="C:/LiqunWA/example-models/Phi-3.5/phi-3.5-mini-4k-instruct-fp32-cpu/model.onnx"
        model_q4_0_path="C:/LiqunWA/example-models/Phi-3.5/phi-3.5-mini-4k-instruct-lamma_cpp_q4_0-cpu/model.onnx"

        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
        algo_config = matmul_4bits_quantizer.LlamaCppQuantConfig(
            quant_type_name="q4_0",
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
        )
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            model=model,
            algo_config=algo_config,
        )

        quant.process()
        quant.model.save_model_to_file(model_q4_0_path, True) # save data to external file

    def llama_cpp_quant_test(
        self,
        model_fp32_path: str,
        data_reader: TestDataFeeds,
        quant_type_name: str,
        op_types_to_quantize: tuple[str, ...] = ("MatMul",),
        quant_axes: tuple[tuple[str, int], ...] = (("MatMul", 0), ("Gather", 1)),
        rtol: float = 0.01,
        atol: float = 0.05,
    ):
        name_prefix = "llama.cpp"
        model_out_path = str(
            Path(self._tmp_model_dir.name).joinpath(f"{name_prefix}_{quant_type_name}.onnx").absolute()
        )

        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
        algo_config = matmul_4bits_quantizer.LlamaCppQuantConfig(
            quant_type_name=quant_type_name,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
        )
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            model=model,
            algo_config=algo_config,
        )

        quant.process()
        quant.model.save_model_to_file(model_out_path, True) # save data to external file

        quant_nodes = {"MatMulNBits": 1}
        check_op_type_count(self, model_out_path, **quant_nodes)

        data_reader.rewind()

        check_model_correctness(self, model_fp32_path, model_out_path, data_reader.get_next(), rtol, atol)

    # @parameterized.expand([
    #     # ("q4_0", 256, 1),
    #     # ("q4_0", 256, 4),
    #     # ("q4_0", 256, 10),
    #     # ("q4_0", 256, 17),
    #     # ("q4_0", 1024, 1),
    #     # ("q4_0", 1024, 4),
    #     # ("q4_0", 1024, 10),
    #     # ("q4_0", 1024, 17),
    #     ("q4_1", 256, 1),
    #     ("q4_1", 256, 4),
    #     # ("q4_1", 256, 10),
    #     # ("q4_1", 256, 17),
    #     # ("q4_1", 1024, 1),
    #     # ("q4_1", 1024, 4),
    #     # ("q4_1", 1024, 10),
    #     # ("q4_1", 1024, 17),
    #     ("tq1_0", 256, 1),
    #     ("tq1_0", 256, 4),
    #     # ("tq1_0", 256, 10),
    #     # ("tq1_0", 256, 17),
    #     # ("tq1_0", 1024, 1),
    #     # ("tq1_0", 1024, 4),
    #     # ("tq1_0", 1024, 10),
    #     # ("tq1_0", 1024, 17),
    #     ("q2_K", 256, 1),
    #     ("q4_K", 256, 1),
    #     ("q5_K", 256, 1),
    #     ("q6_K", 256, 1),
    #     ("q2_K", 256, 2),
    #     ("q4_K", 256, 2),
    #     ("q5_K", 256, 2),
    #     ("q6_K", 256, 2),
    # ])


    # def test_quantize_matmul_llama_cpp(self, quant_type_name, in_features, out_features):
    #     np.random.seed(13)
    #     model_fp32_path = str(
    #         Path(self._tmp_model_dir.name).joinpath(
    #             f"matmul_fp32_{quant_type_name}_{in_features}_{out_features}.onnx"
    #         ).absolute()
    #     )

    #     self.construct_model_matmul(
    #         model_fp32_path, symmetric=True, in_features=in_features, out_features=out_features,
    #         llama_cpp_quant_type_name=quant_type_name
    #     )
    #     data_reader = self.input_feeds(1, {"input": (1, in_features)})

    #     self.llama_cpp_quant_test(
    #         model_fp32_path=model_fp32_path,
    #         data_reader=data_reader,
    #         quant_type_name=quant_type_name,
    #     )

    quant_type_names = [
        "q4_0",
        "q4_1",
        "q5_0",
        "q5_1",
        "q8_0",
        "q2_K",
        "q3_K",
        "q4_K",
        "q5_K",
        "q6_K",
        "tq1_0",
        "tq2_0",
    ]
    in_features_list = [
        256,
        1024]
    out_features_list = [
        1,
        4,
        10,
        17]

    test_cases = list(product(quant_type_names, in_features_list, out_features_list))

    @parameterized.expand(test_cases)
    def test_quantize_matmul_llama_cpp(self, quant_type_name, in_features, out_features):
        np.random.seed(13)
        model_fp32_path = str(
            Path(self._tmp_model_dir.name).joinpath(
                f"matmul_fp32_{quant_type_name}_{in_features}_{out_features}.onnx"
            ).absolute()
        )

        self.construct_model_matmul(
            model_fp32_path, symmetric=True, in_features=in_features, out_features=out_features,
            llama_cpp_quant_type_name=quant_type_name
        )
        data_reader = self.input_feeds(1, {"input": (1, in_features)})

        self.llama_cpp_quant_test(
            model_fp32_path=model_fp32_path,
            data_reader=data_reader,
            quant_type_name=quant_type_name,
        )

if __name__ == "__main__":
    unittest.main()
