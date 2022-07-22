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


class FusionQOrderedSkipLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "QOrderedSkipLayerNormalization", "SkipLayerNormalization")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
                Fuse (quantized) Skip Layer Normalization subgraph into one node QOrderedSkipLayerNormalization:
                        input  -> DQ
                                  |
                                  |
            (other inputs)-> SkipLayerNormalization --> Q -->
                                  |
                                  | 
                      residual -> DQ  
        """

        children = self.model.get_children(node, input_name_to_nodes)

        # Should only have 1 child - QuantizeLinear
        if len(children) != 1 or children[0].op_type != "QuantizeLinear":
            return

        downstream_quantize_node = input_name_to_nodes[node.output[0]][0]
        if downstream_quantize_node.op_type != "QuantizeLinear":
            return

        y_scale = self.model.get_constant_value(downstream_quantize_node.input[1])
        if y_scale is None:
            return

        y_zero_point = self.model.get_constant_value(downstream_quantize_node.input[2])
        if y_zero_point is None or y_zero_point != 0:
            return

        # The first and second inputs to SkipLayerNormalization should flow through DequantizeLinear nodes
        first_path_id, first_input_parent_nodes, _ = self.model.match_parent_paths(
            node,
            [
                (["DequantizeLinear"], [0])
            ],
            output_name_to_node,
        )

        if first_path_id < 0:
            return

        dequantize_node_0 = first_input_parent_nodes[0]

        x_scale_0 = self.model.get_constant_value(dequantize_node_0.input[1])
        if x_scale_0 is None:
            return

        x_zero_point_0 = self.model.get_constant_value(dequantize_node_0.input[2])
        if x_zero_point_0 is None or x_zero_point_0 != 0:
            return

        second_path_id, second_input_parent_nodes, _ = self.model.match_parent_paths(
            node,
            [
                (["DequantizeLinear"], [1])
            ],
            output_name_to_node,
        )

        if second_path_id < 0:
            return

        dequantize_node_1 = second_input_parent_nodes[0]

        x_scale_1 = self.model.get_constant_value(dequantize_node_1.input[1])
        if x_scale_1 is None:
            return

        x_zero_point_1 = self.model.get_constant_value(dequantize_node_1.input[2])
        if x_zero_point_1 is None or x_zero_point_1 != 0:
            return

        subgraph_nodes = [node]  #SkipLayerNormalization
        subgraph_nodes.extend([downstream_quantize_node, dequantize_node_0, dequantize_node_1])  #Relevant Q, DQ nodes

        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            downstream_quantize_node.output,
            input_name_to_nodes,
            output_name_to_node,
        ):
            logger.debug(f"It is not safe to fuse QOrderedSkipLayerNormalization node. Skip")
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        normalize_node = helper.make_node(
            "QOrderedSkipLayerNormalization",
            inputs=[dequantize_node_0.input[0], dequantize_node_0.input[1], 
                    dequantize_node_1.input[0], dequantize_node_1.input[1], 
                    node.input[2], node.input[3], downstream_quantize_node.input[1]],
            outputs=[downstream_quantize_node.output[0]],
            name=self.model.create_node_name("QOrderedSkipLayerNormalization", name_prefix="QOrderedSkipLayerNormalization"),
        )

        #TODO: More attributes
        self.nodes_to_add.append(normalize_node)
        self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name

