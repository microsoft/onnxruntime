# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import onnx

from ...onnx_model import ONNXModel


class Fusion:
    """
    Base class for fusions.
    """

    def __init__(self, model: ONNXModel, fused_op_type: str, search_op_type: str):
        self.search_op_type: str = search_op_type
        self.fused_op_type: str = fused_op_type
        self.model: ONNXModel = model
        self.nodes_to_remove: list = []
        self.nodes_to_add: list = []

    def fuse(self, node: onnx.NodeProto, input_name_to_nodes: dict[str, list[onnx.NodeProto]],
             output_name_to_node: dict[str, onnx.NodeProto]):
        """
        Interface function for derived fusion classes. Tries to fuse a node sequence containing
        the specified node.
        """
        raise NotImplementedError

    def apply(self) -> bool:
        """
        Apply graph fusion on the entire model graph.
        """
        input_name_to_nodes = self.model.input_name_to_nodes()
        output_name_to_node = self.model.output_name_to_node()

        for node in self.model.nodes():
            if node.op_type == self.search_op_type:
                self.fuse(node, input_name_to_nodes, output_name_to_node)

        self.model.remove_nodes(self.nodes_to_remove)
        self.model.add_nodes(self.nodes_to_add)

        graph_updated = bool(self.nodes_to_remove or self.nodes_to_add)

        if graph_updated:
            self.model.remove_unused_constant()

        return graph_updated

    def is_safe_to_fuse_nodes(self, nodes_to_remove: list[onnx.NodeProto],
                              keep_outputs: list[str],
                              input_name_to_nodes: dict[str, list[onnx.NodeProto]],
                              output_name_to_node: dict[str, onnx.NodeProto]):
        for node_to_remove in nodes_to_remove:
            for output_to_remove in node_to_remove.output:
                if output_to_remove in keep_outputs:
                    continue

                if output_to_remove in input_name_to_nodes:
                    for impacted_node in input_name_to_nodes[output_to_remove]:
                        if impacted_node not in nodes_to_remove:
                            # Not safe to remove nodes since output is used by impacted_node
                            return False
        return True

