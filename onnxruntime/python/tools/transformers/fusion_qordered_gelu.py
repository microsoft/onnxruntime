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


class FusionQOrderedGelu(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "QOrderedGelu", ["Gelu", "FastGelu"])

    def fuse(self, node, input_name_to_nodes: Dict, output_name_to_node: Dict):
        """
        INPUT PATTERN
        Fuse (quantized) Gelu subgraph into one node QOrderedGelu:
            -> quantized input  -> DQ -> Gelu -> Q ->

        (or)

            -> quantized input  -> DQ -> FastGelu -> Q ->

        (or)

            -> quantized input  -> DQ -> Reshape -> FastGelu -> Q ->

        OUTPUT PATTERN
            -> QOrderedGelu ->

        (or)
            -> Reshape -> QOrderedGelu ->
        """
        gelu_children = self.model.get_children(node, input_name_to_nodes)

        # Should only have 1 child - QuantizeLinear
        if len(gelu_children) != 1 or gelu_children[0].op_type != "QuantizeLinear":
            return

        downstream_quantize_node = gelu_children[0]

        if not FusionUtils.check_qdq_node_for_fusion(downstream_quantize_node, self.model):
            return

        # The first input to Gelu/FastGelu should flow through a DequantizeLinear node
        
        # In GPT2, there a Reshape between the DQ node and the FastGelu node to reshape
        # the output of the upstream Gemm. In this case, we move the DQ node below the 
        # Reshape node and then fuse the FastGelu node.
        dq_paths = {
            "path1": (["DequantizeLinear"], [0]),
            "path2": (["Reshape", "DequantizeLinear"], [0, 0]),
        }

        dq_path = None
        has_reshape = False
        for k, v in dq_paths.items():
            dq_path = self.model.match_parent_path(dq_paths, v[0], v[1])
            if dq_path is None:
                continue
            if k == "path2":
                has_reshape = True
            break

        if dq_path is None:
            logger.debug(f"It is not safe to fuse QOrderedGelu node. Skip")
            return

        upstream_reshape_node = dq_path[0] if has_reshape else None
        upstream_dequantize_node = dq_path[1] if has_reshape else dq_path[0]

        if not FusionUtils.check_qdq_node_for_fusion(upstream_dequantize_node, self.model):
            return

        # Push Reshape node above the DQ node if it exists
        if has_reshape:
            """
            -> DQ -> Reshape -> FastGelu -> Q ->

            should become

            -> Reshape -> DQ -> FastGelu -> Q ->            
            """
            self.model.replace_node_input(upstream_reshape_node, upstream_reshape_node.input[0], upstream_dequantize_node.input[0])
            self.model.replace_node_input(upstream_dequantize_node, upstream_dequantize_node.input[0], upstream_reshape_node.output[0])
            self.model.replace_node_input(node, node.input[0], upstream_dequantize_node.output[0])

        # Fusion logic
        subgraph_nodes = [node]  # Gelu/FastGelu
        subgraph_nodes.extend([downstream_quantize_node, upstream_dequantize_node])  # Relevant Q, DQ nodes

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
            inputs=[
                upstream_dequantize_node.input[0],
                upstream_dequantize_node.input[1],
                downstream_quantize_node.input[1],
            ],
            outputs=[downstream_quantize_node.output[0]],
            name=self.model.create_node_name("QOrderedGelu", name_prefix="QOrderedGelu"),
        )

        # TODO: We only support CuBlasLt order ORDER_ROW for now.
        # Once we start supporting other data ordering format(s), we
        # will support user configuring the data ordering for the op.
        ordered_gelu_node.attribute.extend([helper.make_attribute("order_X", 1)])
        ordered_gelu_node.attribute.extend([helper.make_attribute("order_Y", 1)])

        ordered_gelu_node.domain = "com.microsoft"

        self.nodes_to_add.append(ordered_gelu_node)
        self.node_name_to_graph_name[ordered_gelu_node.name] = self.this_graph_name
