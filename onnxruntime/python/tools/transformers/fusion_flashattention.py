# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
from logging import getLogger
from os import name
from sys import path
from typing import Tuple, Union

import numpy as np
from fusion_base import Fusion
from fusion_options import AttentionMaskFormat
from fusion_utils import FusionUtils, NumpyHelper
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel
from shape_infer_helper import SymbolicShapeInferenceHelper, get_shape_from_type_proto

logger = getLogger(__name__)


class FusionFlashAttention(Fusion):
    """
    Fuse Flash Attention subgraph into one node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__(model, "FlashAttention", ['Softmax'])
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def create_attention_node(
        self,
        q_input: str,
        k_input: str,
        v_input: str,
        add_input: str,
        mask_input: str,
        output: str,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            q_input (str): input name for Q
            k_input (str): input name for  K
            v_input (str): input name for  V
            add_input (str): Add bias node in fully connection for K
            mask_input (str): Add mask node in fully connection for V
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        # For FlashAttention operator, use separated inputs for query, key and value, and no weights.
        attention_node_name = self.model.create_node_name("FlashAttention")
        attention_node = helper.make_node(
            "FlashAttention",
            inputs=[
                q_input,
                k_input,
                v_input,
                add_input if add_input is not None else "",
                mask_input if mask_input is not None else "",
                ],
            outputs=[output],
            name=attention_node_name,
        )

        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", self.num_heads)])

        return attention_node

    def fuse_alibi_pattern(self, softmax_node, input_name_to_nodes, output_name_to_node):
        """
            K(N*H,L,D)        Alibi(N*H, 1, S)     Mask(N,1,S,L)
            |                   |                    |
        Q-->MatMul --> Mul --> Add --> Reshape --> Add --> [Softmax] --> Reshape --> MatMul --> Transpose
                                                               |
                                                               |
        V-------------------------------------------------------

        """
        need_to_remove = [softmax_node]

        # search nodes before softmax
        qk_matmul_nodes = self.model.match_parent_path(
            softmax_node,
            ["Add", "Reshape", "Mul", "MatMul", "Transpose"],
            [0, None, 0, 0, 1],
        )
        if qk_matmul_nodes is not None:
            (mask_add, r_mask, qk_mul, matmul_qk, trans_k) = qk_matmul_nodes
        else:
            return None, []

        need_to_remove.extend(qk_matmul_nodes)

        # get q and k node
        q_input, k_input = matmul_qk.input[0], trans_k.input[0]

        # get add input 
        #add_input = qk_add.input[0] if qk_add.input[1] == qk_mul.output[0] else qk_add.input[1]
        add_input = None

        # get mask add input
        mask_input = mask_add.input[0] if mask_add.input[1] == r_mask.output[0] else mask_add.input[1]

        # search nodes after softmax: MatMul and Transpose
        sv_matmul = self.model.find_first_child_by_type(softmax_node, "MatMul")
        if sv_matmul is None:
            logger.warn(f'can not find matmal after softmax')
            return None, []

        # check matmul
        reshape_nodes = self.model.match_parent_path(sv_matmul, ['Reshape', 'Softmax'])
        if reshape_nodes is None:
            logger.warn(f'can not find matmal after softmax')
            return None, []

        (r_n, s_n) = reshape_nodes
        if s_n != softmax_node:
            logger.warn(f'can not find matmal after softmax')
            return None, []

        need_to_remove.append(r_n)
        need_to_remove.append(sv_matmul)

        # get v from matmul
        v_input = sv_matmul.input[1]

        ## get flash attention node output
        #trans_node = self.model.find_first_child_by_type(sv_matmul, "Transpose")
        #reshape_nodes = self.model.match_parent_path(trans_node, ['Reshape', 'MatMul'], [0, 0])
        #if reshape_nodes is None:
        #    logger.warn('not find transpose after MatMul')
        #    return

        #rn, mm = reshape_nodes
        #if mm != sv_matmul:
        #    logger.warn('not find transpose after MatMul')
        #need_to_remove.append(rn)

        ## get reshape node after transpose
        #reshape_node = input_name_to_nodes[trans_node.output[0]][0]
        #if reshape_node.op_type != 'Reshape':
        #    logger.warn(f'last node is not reshape: {reshape_node}')
        #    return

        #need_to_remove.append(trans_node)
        #need_to_remove.append(reshape_node)
        #attention_last_node = reshape_node
        attention_last_node = sv_matmul

        new_node = self.create_attention_node(
            q_input,
            k_input,
            v_input,
            add_input,
            mask_input,
            attention_last_node.output[0],
        )
        if new_node is None:
            logger.warn('create new node for flash attention failed')
            return None, []

        return new_node, need_to_remove

    def fuse_norm_pattern(self, softmax_node, input_name_to_nodes, output_name_to_node):
        """
            K                            Mask
            |                             |
        Q-->MatMul --> Div -->  Add --> [Softmax] --> MatMul --> Transpose
                                                               |
                                                               |
        V-------------------------------------------------------

        """
        need_to_remove = [softmax_node]

        # search nodes before softmax
        qk_matmul_nodes = self.model.match_parent_path(
            softmax_node,
            ["Add", "Div", "MatMul"],
            [0, None, 0],
        )

        mask_add, qk_div, matmul_qk = qk_matmul_nodes

        need_to_remove.extend(qk_matmul_nodes)

        # get q and k node
        q_input, k_input = matmul_qk.input[0], matmul_qk.input[1]

        # get mask add input
        mask_input = mask_add.input[0] if mask_add.input[1] == qk_div.output[0] else mask_add.input[1]

        # search nodes after softmax: MatMul and Transpose
        sv_matmul = self.model.find_first_child_by_type(softmax_node, "MatMul")
        if sv_matmul is None:
            logger.warn(f'can not find matmal after softmax')
            return None, []

        # check matmul
        before_matmul_nodes = self.model.match_parent_path(sv_matmul, ['Softmax'])
        if before_matmul_nodes is None or before_matmul_nodes[0] != softmax_node:
            logger.warn(f'can not find matmal after softmax')
            return None, []

        need_to_remove.append(sv_matmul)

        # get v from matmul
        v_input = sv_matmul.input[1]

        attention_last_node = sv_matmul

        new_node = self.create_attention_node(
            q_input,
            k_input,
            v_input,
            None,
            mask_input,
            attention_last_node.output[0],
        )
        if new_node is None:
            logger.warn('create new node for flash attention failed')
            return None, []

        return new_node, need_to_remove


    def fuse(self, softmax_node, input_name_to_nodes, output_name_to_node):
        """
            K(N*H,L,D)        Alibi(N*H, 1, S)     Mask(N,1,S,L)
            |                   |                    |
        Q-->MatMul --> Mul --> Add --> Reshape --> Add --> [Softmax] --> Reshape --> MatMul --> Transpose
                                                               |
                                                               |
        V-------------------------------------------------------


            K                   Mask
            |                    |
        Q-->MatMul --> Div -->  Add --> [Softmax] --> MatMul --> Transpose
                                                               |
                                                               |
        V-------------------------------------------------------

        """
        new_node = None
        need_to_remove = []
        # search nodes before softmax
        qk_matmul_nodes = self.model.match_parent_path(
            softmax_node,
            ["Add", "Reshape", "Mul", "MatMul"],
            [0, None, 0, 0],
        )
        if qk_matmul_nodes is not None:
            new_node, need_to_remove = self.fuse_alibi_pattern(softmax_node, input_name_to_nodes, output_name_to_node)
        else:
            # search nodes before softmax
            qk_matmul_nodes = self.model.match_parent_path(
                softmax_node,
                ["Add", "Div", "MatMul"],
                [0, None, 0],
            )
            if qk_matmul_nodes is not None:
                new_node, need_to_remove = self.fuse_norm_pattern(softmax_node, input_name_to_nodes, output_name_to_node)

        if new_node is None:
            return

        if not self.model.is_safe_to_fuse_nodes(need_to_remove, new_node.output, input_name_to_nodes, output_name_to_node):
            logger.warn('not safe to fuse nodes')
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend(need_to_remove)
        self.prune_graph = True

