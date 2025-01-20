# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionDynamo(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "", ["Gemm"])
        self.count = {
            "Gemm to MatMul + Add": 0
        }

    def apply(self):
        super().apply()
        for k, v in self.count.items():
            if v > 0:
                logger.info(f"{k}: {v}")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        """
        Update TorchDynamo-exported graph to produce ops that have already been pattern
        matched with TorchScript-exported graph
        """
        if node.op_type == "Gemm":
            # Convert Gemm to MatMul + Add for easier pattern matching
            self.gemm_to_matmul_add(node, input_name_to_nodes, output_name_to_node)

    def gemm_to_matmul_add(self, node, input_name_to_nodes, output_name_to_node):
        """
        Replace Gemm with MatMul + Add for easier pattern matching in subsequent fusions
        """
        # Check if weight is initializer data
        weight_proto = self.model.get_initializer(node.input[1])
        if weight_proto is None:
            logger.debug("fuse_dynamo (gemm_to_matmul_add): failed to identify weight input")
            return

        # Check if bias is initializer data (if exists)
        bias_proto = None if len(node.input) < 3 else self.model.get_initializer(node.input[2])
        if bias_proto is None:
            logger.debug("fuse_dynamo (gemm_to_matmul_add): failed to identify bias input")
            return
        has_bias = bias_proto is not None

        # Check that all nodes using weight are Gemm ops that also only use the initializer data as input
        skip = False
        for child_node in input_name_to_nodes[node.input[1]]:
            if not (child_node.op_type == "Gemm" and node.input[1] == child_node.input[1]):
                skip = True
                break
        if skip:
            logger.debug("fuse_dynamo (gemm_to_matmul_add): other non-Gemm nodes use the weight input")
            return

        # Check that all nodes using bias are Gemm ops that also only use the initializer data as input (if exists)
        if has_bias:
            skip = False
            for child_node in input_name_to_nodes[node.input[2]]:
                if not (child_node.op_type == "Gemm" and len(child_node.input) == 3 and node.input[2] == child_node.input[2]):
                    skip = True
                    break
            if skip:
                logger.debug("fuse_dynamo (gemm_to_matmul_add): other non-Gemm nodes use the bias input")
                return

        # Check if weight data is 2D
        weight = NumpyHelper.to_array(weight_proto)
        if len(weight.shape) != 2:
            logger.debug("fuse_dynamo (gemm_to_matmul_add): shape of weight data is not 2D")
            return

        # Check attributes are correct values (alpha/beta = 1, transA/transB = 0)
        for attr in node.attribute:
            if attr.name in {"alpha", "beta"} and attr.f != 1:
                logger.debug(f"fuse_dynamo (gemm_to_matmul_add): expected {attr.name} = 1 but found {attr.name} = {attr.f}")
                return
            if attr.name in {"transA", "transB"} and attr.i != 0:
                logger.debug(f"fuse_dynamo (gemm_to_matmul_add): expected {attr.name} = 0 but found {attr.name} = {attr.i}")
                return

        # Replace Gemm with MatMul + Add (if exists)
        matmul_node = helper.make_node(
            "MatMul",
            inputs=[node.input[0], node.input[1]],
            outputs=[node.output[0] + "_MatMul" if has_bias else node.output[0]],
            name=self.model.create_node_name("MatMul"),
        )
        self.nodes_to_add.append(matmul_node)
        self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name
        # self.increase_counter("MatMul")

        if has_bias:
            add_node = helper.make_node(
                "Add",
                inputs=[matmul_node.output[0], node.input[2]],
                outputs=[node.output[0]],
                name=self.model.create_node_name("Add"),
            )
            self.nodes_to_add.append(add_node)
            self.node_name_to_graph_name[add_node.name] = self.this_graph_name
            # self.increase_counter("Add")

        # Add node to list of nodes to remove
        self.nodes_to_remove.append(node)

        self.count["Gemm to MatMul + Add"] += 1
