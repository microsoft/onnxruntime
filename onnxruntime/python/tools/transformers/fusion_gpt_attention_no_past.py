#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
import numpy as np
from logging import getLogger
from onnx import helper, numpy_helper, TensorProto
from onnx_model import OnnxModel
from fusion_base import Fusion
from fusion_utils import FusionUtils

logger = getLogger(__name__)


class FusionGptAttentionNoPast(Fusion):
    """
    Fuse GPT-2 Attention without past state into one Attention node.
    This does not support attention_mask graph input right now.
    """
    def __init__(self, model: OnnxModel, num_heads: int):
        super().__init__(model, "Attention", "LayerNormalization", "without past")
        # TODO: detect num_heads from graph like FusionAttention
        self.num_heads = num_heads

    def create_attention_node(self, gemm, gemm_qkv, input, output):
        attention_node_name = self.model.create_node_name('Attention')
        attention_node = helper.make_node('Attention',
                                          inputs=[input, gemm.input[1], gemm.input[2]],
                                          outputs=[attention_node_name + "_output"],
                                          name=attention_node_name)
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend(
            [helper.make_attribute("num_heads", self.num_heads),
             helper.make_attribute("unidirectional", 1)])

        matmul_node = helper.make_node('MatMul',
                                       inputs=[attention_node_name + "_output", gemm_qkv.input[1]],
                                       outputs=[attention_node_name + "_matmul_output"],
                                       name=attention_node_name + "_matmul")

        add_node = helper.make_node('Add',
                                    inputs=[attention_node_name + "_matmul_output", gemm_qkv.input[2]],
                                    outputs=[output],
                                    name=attention_node_name + "_add")

        self.nodes_to_add.extend([attention_node, matmul_node, add_node])
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name
        self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name
        self.node_name_to_graph_name[add_node.name] = self.this_graph_name

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        return_indice = []
        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ['Add', 'Reshape', 'Gemm', 'Reshape', 'Reshape', 'Transpose', 'MatMul'],
            [0, None, 0, 0, 0, 0, 0],
            output_name_to_node=output_name_to_node,
            return_indice=return_indice
            ) # yapf: disable
        if qkv_nodes is None:
            return
        (add_qkv, reshape_qkv, gemm_qkv, reshape_1, reshape_2, transpose_qkv, matmul_qkv) = qkv_nodes

        another_input = add_qkv.input[1 - return_indice[0]]

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ['Transpose', 'Reshape', 'Split', 'Reshape', 'Gemm', 'Reshape'],
            [1, 0, 0, 0, 0, 0]) # yapf: disable
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (transpose_v, reshape_v, split_v, reshape_after_gemm, gemm, reshape_before_gemm) = v_nodes

        layernorm_before_attention = self.model.get_parent(reshape_before_gemm, 0, output_name_to_node)
        if layernorm_before_attention is None or layernorm_before_attention.op_type != 'LayerNormalization':
            if layernorm_before_attention.op_type != 'Add':
                logger.debug(f"failed to get layernorm before gemm. Got {layernorm_before_attention.op_type}")
                return

        if not another_input in layernorm_before_attention.input:
            # match openai-gpt
            if not another_input in layernorm_before_attention.output:
                logger.debug("Add and LayerNormalization shall have one same input")
                return

        qk_nodes = self.model.match_parent_path(matmul_qkv, ['Softmax', 'Sub', 'Mul', 'Div', 'MatMul'], [0, 0, 0, 0, 0])
        if qk_nodes is not None:
            (softmax_qk, sub_qk, mul_qk, div_qk, matmul_qk) = qk_nodes
            mask_nodes = self.model.match_parent_path(
                sub_qk,
                ['Mul', 'Sub', 'Slice', 'Slice', 'Unsqueeze', 'Sub', 'Squeeze', 'Slice', 'Shape', 'Div'],
                [1,      0,     1,       0,       1,           0,     0,         0,       0,       0])  # yapf: disable
            if mask_nodes is None:
                logger.debug("fuse_attention: failed to match mask path")
                return
            div_mask = mask_nodes[-1]

            if div_qk != div_mask:
                logger.debug("fuse_attention: skip since div_qk != div_mask")
                return
        else:
            # New pattern for gpt2 from PyTorch 1.5.0 and Transformers 2.9.0.
            qk_nodes = self.model.match_parent_path(matmul_qkv, ['Softmax', 'Where', 'Div', 'MatMul'], [0, 0, 1, 0])
            if qk_nodes is not None:
                (softmax_qk, where_qk, div_qk, matmul_qk) = qk_nodes
                mask_nodes = self.model.match_parent_path(
                    where_qk,
                    ['Cast', 'Slice', 'Slice', 'Unsqueeze', 'Sub', 'Squeeze', 'Slice', 'Shape', 'Div'],
                    [ 0,     0,       0,       1,           0,     0,         0,       0,       0])  # yapf: disable
                if mask_nodes is None:
                    logger.debug("fuse_attention: failed to match mask path")
                    return
                div_mask = mask_nodes[-1]

                if div_qk != div_mask:
                    logger.debug("fuse_attention: skip since div_qk != div_mask")
                    return
            else:
                # match openai-gpt
                qk_nodes = self.model.match_parent_path(matmul_qkv, ['Softmax', 'Add', 'Mul', 'Div', 'MatMul'],
                                                        [0, 0, 0, 0, 0])
                if qk_nodes is None:
                    logger.debug("fuse_attention: failed to match qk path")
                    return
                (softmax_qk, add_qk, mul_qk, div_qk, matmul_qk) = qk_nodes
                mask_nodes = self.model.match_parent_path(
                    mul_qk,
                    ['Slice', 'Slice', 'Unsqueeze', 'Squeeze', 'Slice', 'Shape', 'Div'],
                    [ 1,       0,       2,           0,         0,       0,       0])  # yapf: disable
                if mask_nodes is None:
                    logger.debug("fuse_attention: failed to match mask path")
                    return
                div_mask = mask_nodes[-1]

                if div_qk != div_mask:
                    logger.debug("fuse_attention: skip since div_qk != div_mask")
                    return

        q_nodes = self.model.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Split'], [0, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        (transpose_q, reshape_q, split_q) = q_nodes
        if split_v != split_q:
            logger.debug("fuse_attention: skip since split_v != split_q")
            return

        k_nodes = self.model.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Split'], [1, 0, 0])
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return
        (transpose_k, reshape_k, split_k) = k_nodes
        if split_v != split_k:
            logger.debug("fuse_attention: skip since split_v != split_k")
            return

        self.create_attention_node(gemm, gemm_qkv, layernorm_before_attention.output[0], reshape_qkv.output[0])

        # we rely on prune_graph() to clean old subgraph nodes:
        # qk_nodes + q_nodes + k_nodes + v_nodes + mask_nodes + [reshape_qkv, transpose_qkv, matmul_qkv]
        self.prune_graph = True
