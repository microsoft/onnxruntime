#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

from logging import getLogger
from typing import Dict

from numpy import transpose

from fusion_base import Fusion
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionQOrderedGelu(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "QOrderedGelu", "Gelu")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        gelu_children = self.model.get_children(node, input_name_to_nodes)

        # Should only have 1 child - QuantizeLinear
        if len(gelu_children) != 1 or gelu_children[0].op_type != "QuantizeLinear":
            return

        downstream_quantize_node = gelu_children[0]
 
        # Make sure the downstream QuantizeLinear has the proper zero points and scales    
        y_scale = self.model.get_constant_value(downstream_quantize_node.input[1])
        if y_scale is None:
            return

        y_zero_point = self.model.get_constant_value(downstream_quantize_node.input[2])
        if y_zero_point is None or y_zero_point != 0:
            return

        # The first input to Gelu should flow through a DequantizeLinear node
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

        # Make sure the upstream DequantizeLinear has the proper zero points and scales   
        x_scale_0 = self.model.get_constant_value(dequantize_node_0.input[1])
        if x_scale_0 is None:
            return

        x_zero_point_0 = self.model.get_constant_value(dequantize_node_0.input[2])
        if x_zero_point_0 is None or x_zero_point_0 != 0:
            return

        # Fusion logic            
        subgraph_nodes = [node]  #Gelu
        subgraph_nodes.extend([downstream_quantize_node, dequantize_node_0])  #Relevant Q, DQ nodes

        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            downstream_quantize_node.output,
            input_name_to_nodes,
            output_name_to_node,
        ):
            logger.debug(f"It is not safe to fuse QOrderedGelu node. Skip")
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        ordered_gelu_node = helper.make_node(
            "QOrderedGelu",
            inputs=[dequantize_node_0.input[0], dequantize_node_0.input[1],
                    downstream_quantize_node.input[1]],
            outputs=[downstream_quantize_node.output[0]],
            name=self.model.create_node_name("QOrderedGelu", name_prefix="QOrderedGelu"),
        )

        # TODO 3: More attributes
        self.nodes_to_add.append(ordered_gelu_node)
        self.node_name_to_graph_name[ordered_gelu_node.name] = self.this_graph_name
