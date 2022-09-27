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


class FusionMatmulBiasGelu(Fusion):
    def __init__(self, model: OnnxModel, is_fastgelu):
        super().__init__(model, "MatmulBiasGelu", "BiasGelu")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        matmul = self.model.match_parent_path(node, ["MatMul"], [0])
        
        if matmul is None:
            return

        subgraph_nodes = [matmul, node]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes, [node.output[0]], input_name_to_nodes, output_name_to_node
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        fused_node = helper.make_node(
            "MatmulBiasGelu",
            inputs=[matmul.input[0], matmul.input[1], node.input[1]],
            outputs=node.output,
            name=self.model.create_node_name("MatmulBiasGelu", "BiasGelu_AddBias_"),
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
