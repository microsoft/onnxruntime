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
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime import get_available_providers
from onnxruntime.quantization import quant_utils


@unittest.skipIf(
    "CUDAExecutionProvider" not in get_available_providers(), reason="CUDA is not available, skipping tests."
)
class TestOpMatMul8Bits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_model_dir = tempfile.TemporaryDirectory(prefix="test_matmul8bits.")

    @classmethod
    def tearDownClass(cls):
        cls._tmp_model_dir.cleanup()

    def fill_weight_data(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.random.normal(0, 0.01, size=shape).astype(np.float32)

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

    def construct_model_matmul(self, output_model_path: str, k: int = 32, n: int = 64) -> None:
        """Create a simple onnx model with one MatMul node like (input) --> MatMul --> (output)."""
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_matmul(
            input_name, weight_shape: int | tuple[int, ...], weight_name: str, output_name: str, node_name: str
        ):
            weight_data = self.fill_weight_data(weight_shape)
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
        input_tensor = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, [-1, in_features])
        output_tensor = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [-1, out_features])
        graph_name = "matmul_8bits_test"
        graph = onnx.helper.make_graph(
            [matmul_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        # blocked quantization requires DQ op set >= 21
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 21)])
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
        config: str = "default",
        suffix: str = "",
    ):
        use_qdq = quant_format == quant_utils.QuantFormat.QDQ
        name_prefix = "QDQ" if use_qdq else "QOperator"
        model_int8_path = str(
            Path(self._tmp_model_dir.name)
            .joinpath(f"{name_prefix}_bs{block_size}_{is_symmetric}{suffix}.onnx")
            .absolute()
        )

        # Quantize fp32 model to int8 model
        from onnxruntime.quantization import matmul_nbits_quantizer  # noqa: PLC0415

        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))

        assert config in ["default", "hqq"]
        if config == "default":
            quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=block_size,
                is_symmetric=is_symmetric,
                quant_format=quant_format,
                op_types_to_quantize=op_types_to_quantize,
                quant_axes=quant_axes,
                bits=8,
            )
        else:
            quant_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
                block_size=block_size,
                bits=8,
                quant_format=quant_format,
                op_types_to_quantize=op_types_to_quantize,
                quant_axes=quant_axes,
            )

        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(model, bits=8, algo_config=quant_config)
        quant.process()
        quant.model.save_model_to_file(model_int8_path, False)

        if "Gather" in op_types_to_quantize:
            quant_nodes = {"GatherBlockQuantized": 1}
        else:
            quant_nodes = {"DequantizeLinear": 1, "MatMul": 1} if use_qdq else {"MatMulNBits": 1}
        check_op_type_count(self, model_int8_path, **quant_nodes)

        if use_qdq:
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
            check_qtype_by_node_type(self, model_int8_path, dqnode_io_qtypes)
            for op in quant.model.opset_import():
                if op.domain in [None, "", "ai.onnx"] and op.version < 21:
                    self.fail(f"In QDQ format {op.domain} opset should be >= 21")

        data_reader.rewind()

        try:
            check_model_correctness(
                self,
                model_fp32_path,
                model_int8_path,
                data_reader.get_next(),
                rtol,
                atol,
                providers=["CUDAExecutionProvider"],
            )
        except Exception as exception:
            if "8b quantization not yet supported on this hardware platform!" in exception.args[0]:
                # Currently we don't have int8 quantization support on all platforms, has to tolerate this exception
                pass
            else:
                raise exception

    def test_quantize_matmul_8bits(self):
        np.random.seed(13)
        for k in [32, 40, 256, 512, 512, 1024, 1040]:
            for n in [8, 256]:
                model_fp32_path = str(
                    Path(self._tmp_model_dir.name).joinpath(f"matmul_fp32_k_{k}_n_{n}.onnx").absolute()
                )
                self.construct_model_matmul(model_fp32_path, k=k, n=n)
                for m in [1, 2]:
                    data_reader = self.input_feeds(m, {"input": (m, k)})
                    for config in ["default", "hqq"]:
                        for block_size in [16, 128, 256]:
                            if block_size <= k:
                                self.quant_test(
                                    model_fp32_path,
                                    data_reader,
                                    block_size,
                                    True,
                                    atol=0.01,
                                    rtol=0.01,
                                    config=config,
                                    suffix=f"_m_{m}_n_{n}_k_{k}",
                                )


if __name__ == "__main__":
    unittest.main()
