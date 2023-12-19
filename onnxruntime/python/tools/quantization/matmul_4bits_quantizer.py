# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import onnx
from onnx.onnx_pb import GraphProto, ModelProto, NodeProto, TensorProto

from onnxruntime.capi._pybind_state import quantize_matmul_4bits

from .onnx_model import ONNXModel
from .quant_utils import attribute_to_kwarg

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class MatMul4BitsQuantizer:
    """Perform 4b quantization of constant MatMul weights"""

    def __init__(self, model: ModelProto, block_size: int, is_symmetric: bool, nodes_to_exclude=None):
        if nodes_to_exclude is None:
            nodes_to_exclude = []
        self.model = ONNXModel(model)
        self.block_size = block_size
        self.is_symmetric = is_symmetric
        self.nodes_to_exclude = set(nodes_to_exclude)

    @staticmethod
    def __get_initializer(name, graph_path: List[GraphProto]) -> Tuple[TensorProto, GraphProto]:
        for gid in range(len(graph_path) - 1, -1, -1):
            graph = graph_path[gid]
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor, graph
        return None, None

    def int4_block_quant(self, fp32weight: npt.ArrayLike) -> np.ndarray:
        """4b quantize fp32 weight to a blob"""

        if len(fp32weight.shape) != 2:
            raise ValueError("Current int4 block quantization only supports 2D tensors!")
        rows, cols = fp32weight.shape

        block_size = self.block_size
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        padded_rows = k_blocks * block_size
        pad_len = padded_rows - rows
        if pad_len > 0:
            fp32weight = np.pad(fp32weight, ((0, pad_len), (0, 0)), "constant")

        # block wise quantization, each block comes from a single column
        packed = np.zeros((cols, k_blocks, blob_size), dtype="uint8")
        scales = np.zeros((cols * k_blocks), dtype=fp32weight.dtype)
        zero_point = np.zeros(cols * ((k_blocks + 1) // 2), dtype="uint8")
        quantize_matmul_4bits(packed, fp32weight, scales, zero_point, block_size, cols, rows, self.is_symmetric)

        return (packed, scales, zero_point)

    def _q4_matmul_node_weight(self, node: NodeProto, graph_stack: List[GraphProto]) -> NodeProto:
        """If the node is MatMul with fp32 const weight, quantize the weight with int4, and return the new node"""

        if node.op_type != "MatMul":
            return node  # only care about MatMul for now

        logger.info(f"start to quantize {node.name} ...")
        if node.name in self.nodes_to_exclude:
            logger.info(f"exclude to quantize {node.name} as specified by nodes_to_exclude...")
            return node

        inputB = node.input[1]  # noqa: N806
        B, Bs_graph = MatMul4BitsQuantizer.__get_initializer(inputB, graph_stack)  # noqa: N806
        if B is None:
            logger.info("MatMul doesn't have const weight. Skip to quantize")
            return node  # only care about constant weight

        B_array = onnx.numpy_helper.to_array(B)  # noqa: N806
        if len(B_array.shape) != 2:
            logger.info("MatMul weight is not 2D. Skip to quantize")
            return node  # can only process 2-D matrix

        packed, scales, zero_points = self.int4_block_quant(B_array)
        B_quant = onnx.numpy_helper.from_array(packed)  # noqa: N806
        B_quant.name = B.name + "_Q4"
        for input in Bs_graph.input:
            if input.name == inputB:
                Bs_graph.input.remove(input)
                break

        scales_tensor = onnx.numpy_helper.from_array(scales)
        scales_tensor.name = B.name + "_scales"
        Bs_graph.initializer.extend([B_quant, scales_tensor])

        input_names = [node.input[0], B_quant.name, scales_tensor.name]
        if not self.is_symmetric:
            zp_tensor = onnx.numpy_helper.from_array(zero_points)
            zp_tensor.name = B.name + "_zero_points"
            Bs_graph.initializer.extend([zp_tensor])
            input_names.append(zp_tensor.name)

        kwargs = {}
        rows, cols = B_array.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = 4
        kwargs["block_size"] = self.block_size

        matmul_q4_node = onnx.helper.make_node(
            "MatMulNBits",
            inputs=input_names,
            outputs=[node.output[0]],
            name=node.name + "_Q4" if node.name else "",
            domain="com.microsoft",
            **kwargs,
        )

        logger.info(f"complete quantization of {node.name} ...")

        return matmul_q4_node

    def _process_subgraph(self, graph_stack: List[GraphProto]):
        new_nodes = []
        graph = graph_stack[-1]

        for node in graph.node:
            graph_attrs = [
                attr
                for attr in node.attribute
                if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS
            ]
            if len(graph_attrs):
                kwargs = {}
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        # recursive call to take care of sub-graph
                        graph_stack.append(attr.g)
                        kv = {attr.name: self._process_subgraph(graph_stack)}
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        value = []
                        for subgraph in attr.graphs:
                            # recursive call to take care of sub-graph
                            graph_stack.append(subgraph)
                            value.extend([self._process_subgraph(graph_stack)])
                        kv = {attr.name: value}
                    else:
                        kv = attribute_to_kwarg(attr)
                    kwargs.update(kv)
                node = onnx.helper.make_node(  # noqa: PLW2901
                    node.op_type, node.input, node.output, name=node.name, **kwargs
                )

            new_nodes.append(self._q4_matmul_node_weight(node, graph_stack))

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        graph_stack.pop()
        return graph

    def process(self):
        # use a stack to keep track of sub-graphs
        graph_stack = [self.model.graph()]
        opset_import = self.model.opset_import()

        has_ms_domain = False
        for opset in opset_import:
            if opset.domain == "com.microsoft":
                has_ms_domain = True
        if not has_ms_domain:
            opset_import.extend([onnx.helper.make_opsetid("com.microsoft", 1)])

        self._process_subgraph(graph_stack)
        self.model.clean_initializers()


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Blockwise int4 quantization for MatMul 2D weight matrices.

A weight matrix is partitioned into into blocks, where each block is a
continguous subset inside each column. Each block is quantized into a
set of 4b integers with a scaling factor and an optional offset.
"""
    )

    parser.add_argument("--input_model", required=True, help="Path to the input model file")
    parser.add_argument("--output_model", required=True, help="Path to the output model file")
    parser.add_argument("--block_size", required=False, default=32, type=int, help="Block size for quantization")
    parser.add_argument(
        "--symmetric", required=False, default=True, type=bool, help="Indicate whether to quantize the model symmetrically"
    )
    parser.add_argument("-v", "--verbose", required=False, action="store_true")
    parser.set_defaults(verbose=False)
    parser.add_argument(
        "--nodes_to_exclude",
        nargs="+",
        type=str,
        required=False,
        default=[],
        help="Specify the nodes to be excluded from quantization with node names",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    input_model_path = args.input_model
    output_model_path = args.output_model

    if os.path.exists(output_model_path):
        logger.error(f"file {output_model_path} already exists")
        raise Exception(f"file {output_model_path} already exists")

    model = onnx.load(input_model_path)
    quant = MatMul4BitsQuantizer(model, args.block_size, args.symmetric, nodes_to_exclude=args.nodes_to_exclude)
    quant.process()
    quant.model.save_model_to_file(output_model_path, True)
