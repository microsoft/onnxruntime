# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
from onnx.onnx_pb import GraphProto, ModelProto, NodeProto, TensorProto

from .onnx_model import ONNXModel
from .q4dq_wrapper import Q4dqWrapper
from .quant_utils import attribute_to_kwarg, load_model_with_shape_infer


class MatMulWeight4Quantizer:
    """Perform 4b quantization of constant MatMul weights"""

    ##################
    # quantization types, must be consistent with native code type
    # MLAS_BLK_QUANT_TYPE defined in mlas_q4.h

    # 32 number block, symmetric quantization, with one fp32 as scale, zero point is always 0
    BlkQ4Sym = 0

    # 32 number block, quantization, with one fp32 as scale, one uint8 zero point
    BlkQ4Zp8 = 1

    def __init__(self, model: ModelProto, q4dq: Q4dqWrapper, quant_type: int):
        self.model = ONNXModel(model)
        self.q4dq = q4dq
        self.quant_type = quant_type

    @staticmethod
    def __get_initializer(name, graph_path: List[GraphProto]) -> Tuple[TensorProto, GraphProto]:
        for gid in range(len(graph_path) - 1, -1, -1):
            graph = graph_path[gid]
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor, graph
        return None, None

    def _q4_matmul_node_weight(self, node: NodeProto, graph_stack: List[GraphProto]) -> NodeProto:
        """If the node is MatMul with fp32 const weight, quantize the weight with int4, and return the new node"""

        if node.op_type != "MatMul":
            return node  # only care about MatMul for now

        inputB = node.input[1]  # noqa: N806
        B, Bs_graph = MatMulWeight4Quantizer.__get_initializer(inputB, graph_stack)  # noqa: N806
        if B is None:
            return node  # only care about constant weight

        # TODO!! assume B is not used by any other node
        B_array = onnx.numpy_helper.to_array(B)  # noqa: N806
        if len(B_array.shape) != 2:
            return node  # can only process 2-D matrix

        rows, cols = B_array.shape
        packed = self.q4dq.quantize(B_array, self.quant_type)

        B_quant = onnx.numpy_helper.from_array(packed)  # noqa: N806
        B_quant.name = B.name + "_Q4"
        Bs_graph.initializer.remove(B)
        for input in Bs_graph.input:
            if input.name == inputB:
                Bs_graph.input.remove(input)
                break

        B_shape = onnx.numpy_helper.from_array(np.array([rows, cols]).astype(np.int64))  # noqa: N806
        B_shape.name = B.name + "_shape"
        Bs_graph.initializer.extend([B_quant, B_shape])

        kwargs = {}
        kwargs["blk_quant_type"] = self.quant_type
        matmul_q4_node = onnx.helper.make_node(
            "MatMulFpQ4",
            inputs=[node.input[0], B_quant.name, B_shape.name],
            outputs=[node.output[0]],
            name=node.name + "_Q4" if node.name else "",
            domain="com.microsoft",
            **kwargs,
        )
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
                    elif attr.type == onnx.AttributeProto.GRAPH:
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
    parser.add_argument(
        "--quant_bin_path",
        required=True,
        help="""Currently quantization code is implemented in a separate binary
(onnxruntime_mlas_q4dq) that is compiled with Onnxruntime native code.
Path to this binary needs to be provided here.""",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_model_path = args.input_model
    output_model_path = args.output_model
    q4dq_bin_path = args.quant_bin_path

    q4dq = Q4dqWrapper(q4dq_bin_path)

    model = load_model_with_shape_infer(Path(input_model_path))
    quant = MatMulWeight4Quantizer(model, q4dq, 0)
    quant.process()
    quant.model.save_model_to_file(output_model_path, False)
