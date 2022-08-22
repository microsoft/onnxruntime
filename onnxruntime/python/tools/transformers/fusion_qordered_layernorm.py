# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from typing import Dict

from fusion_base import Fusion
from fusion_utils import FusionUtils
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionQOrderedLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "QOrderedLayerNormalization", "LayerNormalization")

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
        Fuse (quantized) Layer Normalization subgraph into one node QOrderedLayerNormalization:
            quantized input  -> DQ
                                |
                                |
            (other inputs)-> LayerNormalization --> Q -->

            should become

            (quantized input + other inputs)->  QOrderedLayerNormalization --> Q -->
        """

        children = self.model.get_children(node, input_name_to_nodes)

        # Should only have 1 child - QuantizeLinear
        if len(children) != 1 or children[0].op_type != "QuantizeLinear":
            return

        downstream_quantize_node = children[0]

        if not FusionUtils.check_qdq_node_for_fusion(downstream_quantize_node, self.model):
            return

        # The first input to LayerNormalization should flow through a DequantizeLinear node
        first_path_id, first_input_parent_nodes, _ = self.model.match_parent_paths(
            node,
            [(["DequantizeLinear"], [0])],
            output_name_to_node,
        )

        if first_path_id < 0:
            return

        upstream_dequantize_node = first_input_parent_nodes[0]

        if not FusionUtils.check_qdq_node_for_fusion(upstream_dequantize_node, self.model):
            return

        # Fusion logic
        subgraph_nodes = [node]  # LayerNormalization
        subgraph_nodes.extend([downstream_quantize_node, upstream_dequantize_node])  # Relevant Q, DQ nodes

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
            inputs=[
                upstream_dequantize_node.input[0],
                upstream_dequantize_node.input[1],
                node.input[1],
                node.input[2],
                downstream_quantize_node.input[1]
            ],
            outputs=[downstream_quantize_node.output[0]],
            name=self.model.create_node_name("QOrderedLayerNormalization", name_prefix="QOrderedLayerNormalization"),
        )

        # TODO: We only support CuBlasLt order ORDER_ROW for now.
        # Once we start supporting other data ordering format(s), we
        # will support user configuring the data ordering for the op.
        normalize_node.attribute.extend([helper.make_attribute("order_X", 1)])
        normalize_node.attribute.extend([helper.make_attribute("order_Y", 1)])

        normalize_node.domain = "com.microsoft"

        self.nodes_to_add.append(normalize_node)
        self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name
