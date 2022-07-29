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
        super().__init__(model, "QOrderedAttention", "QOrderedSkipLayerNormalization")

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):

        # QOrderedSkipLayerNormalization has two inputs, and one of them is the root input for attention

        # QKV nodes
        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["QuantizeLinear", "Add", "MatMul", "DequantizeLinear", 
             "QuantizeLinear", "Reshape", "Transpose", "MatMul"],
            [None, 0, None, 0, 0, 0, 0, 0],
        )

        if qkv_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match qkv path")
            return

        (_, _, _, dequantize_qkv, quantize_qkv, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        other_inputs = []
        for i, input in enumerate(normalize_node.input):
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
                                              ["DequantizeLinear", "QuantizeLinear", 
                                               "Transpose", "Reshape", "Add", "MatMul",
                                               "DequantizeLinear", "QuantizeLinear"], 
                                               [1, 0, 0, 0, 0, None, 0, 0])
        if v_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match v path")
            return

        matmul_v = v_nodes[-3]
        add_v = v_nodes[-4]

        # QK nodes
        qk_nodes = self.model.match_parent_path(matmul_qkv, 
                                              ["DequantizeLinear", "QuantizeLinear", 
                                               "Softmax", "Add", "Div", "MatMul"], 
                                               [0, 0, 0, 0, None, 0])
        if qk_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match qk path")
            return

        matmul_qk = qk_nodes[-1]
        add_qk = qk_nodes[-3]

        # Q nodes
        q_nodes = self.model.match_parent_path(matmul_qk, ["DequantizeLinear", "QuantizeLinear",
                                                           "Transpose", "Reshape", "Add", "MatMul",
                                                           "DequantizeLinear", "QuantizeLinear"], 
                                                           [0, 0, 0, 0, 0, None, 0, 0])         

        if q_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match q path")
            return

        matmul_q = q_nodes[-3]
        add_q = q_nodes[-4]

        # K nodes
        k_nodes = self.model.match_parent_path(matmul_qk, ["DequantizeLinear", "QuantizeLinear",
                                                           "Transpose", "Reshape", "Add", "MatMul",
                                                           "DequantizeLinear", "QuantizeLinear"], 
                                                           [1, 0, 0, 0, 0, None, 0, 0])

        if k_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match k path")
            return

        matmul_k = k_nodes[-3]
        add_k = k_nodes[-4]

        # Mask nodes
        mask_nodes =  self.model.match_parent_path(add_qk, ["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
                                                           [None, 0, 1, 0, 0])

        if mask_nodes is None:
            logger.debug("fuse_qordered_attention: failed to match mask_nodes path")
            return