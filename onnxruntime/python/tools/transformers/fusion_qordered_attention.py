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


class FusionQOrderedAttention(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "QOrderedAttention", "QOrderedLayerNormalization")

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # TODO: General comment - make sure all zero points are 0s
        
        add_before_layernorm = self.model.match_parent(normalize_node, "QuantizeLinear", 0)
        if add_before_layernorm is not None:
            start_node = add_before_layernorm
        else:
            return

        # Input QDQ nodes
        input_dq_node = self.model.match_parent_path(
            start_node,
            ["DequantizeLinear"],
            [None],
        )

        if input_dq_node is None:
            logger.debug("fuse_qordered_attention: failed to match input qdq nodes path")
            return

        input_dq_node = input_dq_node[-1]

        # QKV nodes
        qkv_nodes = self.model.match_parent_path(
            start_node,
            ["Add", "MatMul", "Reshape", "Transpose", 
            "DequantizeLinear", "QuantizeLinear", "MatMul"],
            [None, None, None, 0, 0, 0, 0],
        )

        if qkv_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match qkv path")
            return

        (_, _, reshape_qkv, transpose_qkv, dequantize_qkv, quantize_qkv,  matmul_qkv) = qkv_nodes

        other_inputs = []
        for i, input in enumerate(start_node.input):
            if input not in output_name_to_node:
                continue

            if input == qkv_nodes[0].output[0]:
                continue

            other_inputs.append(input)

        if len(other_inputs) != 1:
            return

        root_input = other_inputs[0]

        # V nodes
        v_nodes = self.model.match_parent_path(matmul_qkv, 
                                              ["Transpose", "Reshape", 
                                              "DequantizeLinear", "QuantizeLinear",
                                              "Add", "MatMul"], 
                                               [1, 0, 0, 0, 0, None])
        if v_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match v path")
            return

        matmul_v = v_nodes[-1]
        add_v = v_nodes[-2]

        # QK nodes
        qk_nodes = self.model.match_parent_path(matmul_qkv, 
                                              ["Softmax", "Add", "Div", "MatMul"], 
                                               [0, 0, None, 0])
        if qk_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match qk path")
            return

        matmul_qk = qk_nodes[-1]
        add_qk = qk_nodes[-3]

        # Q nodes
        q_nodes = self.model.match_parent_path(matmul_qk, 
                                              ["Transpose", "Reshape", 
                                              "DequantizeLinear", "QuantizeLinear",
                                              "Add", "MatMul"], 
                                               [0, 0, 0, 0, 0, None])       

        if q_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match q path")
            return

        matmul_q = q_nodes[-1]
        add_q = q_nodes[-2]
        reshape_q = q_nodes[-5]

        # K nodes
        k_nodes = self.model.match_parent_path(matmul_qk, 
                                              ["Transpose", "Reshape", 
                                              "DequantizeLinear", "QuantizeLinear",
                                              "Add", "MatMul"], 
                                               [1, 0, 0, 0, 0, None])       

        if k_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match k path")
            return

        matmul_k = k_nodes[-1]
        add_k = k_nodes[-2]

        # Mask nodes
        mask_nodes =  self.model.match_parent_path(add_qk, ["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
                                                           [None, 0, 1, 0, 0])

        if mask_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match mask_nodes path")
            return


        # Form QOrderedAttention node
        if matmul_v.input[0] == root_input and matmul_q.input[0] == root_input and matmul_k.input[0] == root_input:
            mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])

            num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q)        
        
            if hidden_size > 0 and (hidden_size % num_heads) != 0:
                logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
                return None

            # Formulate the inputs      
            # Actual quantized input 
            attention_inputs = [input_dq_node.input[0]]
            attention_inputs.append(input_dq_node.input[1])

            # Mask input
            if mask_index is not None:
                attention_inputs.append(mask_index)
            else:
                attention_inputs.append("")

            # Name and create Attention node
            attention_node_name = self.model.create_node_name("Attention")

            attention_node = helper.make_node(
                "Attention",
                inputs=attention_inputs,
                outputs=[reshape_qkv.output[0]],
                name=attention_node_name,
            )
            
            attention_node.domain = "com.microsoft"

            # TODO: Adjust Q, K, and V Q->DQ nodes

            # TODO: More attributes
            attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

            self.nodes_to_add.append(attention_node)
            self.node_name_to_graph_name[attention_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([reshape_qkv, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)
            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes)

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            # self.nodes_to_remove.extend(mask_nodes)
            self.prune_graph = True                                