# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from typing import Dict

from fusion_base import Fusion
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionQOrderedLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "QOrderedLayerNormalization", "LayerNormalization")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
                Fuse (quantized) Layer Normalization subgraph into one node QOrderedLayerNormalization:
                        input  -> DQ
                                  |
                                  |
            (other inputs)->  LayerNormalization --> Q -->
        """

        children = self.model.get_children(node, input_name_to_nodes)

        # Should only have 1 child - QuantizeLinear
        if len(children) != 1 or children[0].op_type != "QuantizeLinear":
            return

        downstream_quantize_node = children[0]

        # Make sure the downstream QuantizeLinear has the proper zero points and scales   
        y_scale = self.model.get_constant_value(downstream_quantize_node.input[1])
        if y_scale is None:
            return

        y_zero_point = self.model.get_constant_value(downstream_quantize_node.input[2])
        if y_zero_point is None or y_zero_point != 0:
            return

        # The first input to LayerNormalization should flow through a DequantizeLinear node
        path_id, parent_nodes, _ = self.model.match_parent_paths(
            node,
            [
                (["DequantizeLinear"], [0])
            ],
            output_name_to_node,
        )

        if path_id < 0:
            return

        upstream_dequantize_node = parent_nodes[0]

        # Make sure the upstream DequantizeLinear has the proper zero points and scales   
        x_scale_0 = self.model.get_constant_value(upstream_dequantize_node.input[1])
        if x_scale_0 is None:
            return

        x_zero_point_0 = self.model.get_constant_value(upstream_dequantize_node.input[2])
        if x_zero_point_0 is None or x_zero_point_0 != 0:
            return

        # Fusion logic
        subgraph_nodes = [node]  #LayerNormalization
        subgraph_nodes.extend([downstream_quantize_node, upstream_dequantize_node])  #Relevant Q, DQ nodes

        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            downstream_quantize_node.output,
            input_name_to_nodes,
            output_name_to_node,
        ):
            logger.debug(f"It is not safe to fuse QOrderedLayerNormalization node. Skip")
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        normalize_node = helper.make_node(
            "QOrderedLayerNormalization",
            inputs=[upstream_dequantize_node.input[0], upstream_dequantize_node.input[1],
                    node.input[1], node.input[2], 
                    downstream_quantize_node.input[1]],
            outputs=[downstream_quantize_node.output[0]],
            name=self.model.create_node_name("QOrderedLayerNormalization", 
            name_prefix="QOrderedLayerNormalization"),
        )

        normalize_node.domain = "com.microsoft"
        self.nodes_to_add.append(normalize_node)
        self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name

