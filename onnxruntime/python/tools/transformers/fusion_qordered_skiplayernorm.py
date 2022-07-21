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

        quantize_node = input_name_to_nodes[node.output[0]][0]
        if quantize_node.op_type != "QuantizeLinear":
            return

        # The first and second inputs should flow through DequantizeLinears
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

        # TODO Make sure zero point(s) are scalar 0s for Q/DQ nodes

        subgraph_nodes = [node]  #SkipLayerNormalization
        subgraph_nodes.extend([quantize_node, dequantize_node_0, dequantize_node_1])  #Relevant Q, DQ nodes

        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            quantize_node.output,
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
                    node.input[2], node.input[3], quantize_node.input[1]],
            outputs=[quantize_node.output[0]],
            name=self.model.create_node_name("QOrderedSkipLayerNormalization", name_prefix="QOrderedSkipLayerNormalization"),
        )

        normalize_node.attribute.extend([helper.make_attribute("epsilon", float(add_weight))])
        #TODO: More attributes
        self.nodes_to_add.append(normalize_node)
        self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name

