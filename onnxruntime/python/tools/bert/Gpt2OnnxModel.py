#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
import logging
import onnx
import sys
import argparse
import numpy as np
from collections import deque
from onnx import ModelProto, TensorProto, numpy_helper
from BertOnnxModel import BertOnnxModel

logger = logging.getLogger(__name__)


class Gpt2OnnxModel(BertOnnxModel):

    def __init(self, model, num_heads, hidden_size, sequence_length, input_int32, float16, gpu_only):
        super().__init__(model, num_heads, hidden_size, sequence_length, input_int32, float16, gpu_only)

    def fuse_attention(self):
        """
        Fuse Attention subgraph into one Attention node.
        """
        logger.debug(f"start attention fusion...")

        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        attention_count = 0

        for normalize_node in self.get_nodes_by_op_type("LayerNormalization"):
            return_indice = []
            qkv_nodes = self.match_parent_path(
                normalize_node,
                ['Add', 'Reshape', 'Gemm', 'Reshape', 'Reshape', 'Transpose', 'MatMul'],
                [0, None, 0, 0, 0, 0, 0],
                output_name_to_node=output_name_to_node,
                return_indice=return_indice
                ) # yapf: disable
            if qkv_nodes is None:
                continue
            (add_qkv, reshape_qkv, gemm_qkv, reshape_1, reshape_2, transpose_qkv, matmul_qkv) = qkv_nodes

            another_input = add_qkv.input[1 - return_indice[0]]

            v_nodes = self.match_parent_path(
                matmul_qkv,
                ['Transpose', 'Reshape', 'Split', 'Reshape', 'Gemm', 'Reshape'],
                [1, 0, 0, 0, 0, 0]) # yapf: disable
            if v_nodes is None:
                logger.debug("fuse_attention: failed to match v path")
                continue
            (transpose_v, reshape_v, split_v, reshape_after_gemm, gemm, reshape_before_gemm) = v_nodes

            layernorm_before_attention = self.get_parent(reshape_before_gemm, 0, output_name_to_node)
            if layernorm_before_attention is None or layernorm_before_attention.op_type != 'LayerNormalization':
                logger.debug(f"failed to get layernorm before gemm. Got {layernorm_before_attention.op_type}")
                continue

            if not another_input in layernorm_before_attention.input:
                logger.debug("Add and LayerNormalization shall have one same input")
                continue

            qk_nodes = self.match_parent_path(matmul_qkv, ['Softmax', 'Sub', 'Mul', 'Div', 'MatMul'], [0, 0, 0, 0, 0])
            if qk_nodes is None:
                logger.debug("fuse_attention: failed to match qk path")
                continue
            (softmax_qk, sub_qk, mul_qk, div_qk, matmul_qk) = qk_nodes

            q_nodes = self.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Split'], [0, 0, 0])
            if q_nodes is None:
                logger.debug("fuse_attention: failed to match q path")
                continue
            (transpose_q, reshape_q, split_q) = q_nodes
            if split_v != split_q:
                logger.debug("fuse_attention: skip since split_v != split_q")
                continue

            k_nodes = self.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Split'], [1, 0, 0])
            if k_nodes is None:
                logger.debug("fuse_attention: failed to match k path")
                continue
            (transpose_k, reshape_k, split_k) = k_nodes
            if split_v != split_k:
                logger.debug("fuse_attention: skip since split_v != split_k")
                continue

            mask_nodes = self.match_parent_path(
                sub_qk,
                ['Mul', 'Sub', 'Slice', 'Slice', 'Unsqueeze', 'Sub', 'Squeeze', 'Slice', 'Shape', 'Div'],
                [1,      0,     1,       0,       1,           0,     0,         0,       0,       0])  # yapf: disable
            if mask_nodes is None:
                logger.debug("fuse_attention: failed to match mask path")
                continue
            (mul_mask, sub_mask, slice_mask, slice_mask_0, unsqueeze_mask, sub_mask, squeeze_mask, slice_mask_1,
             shape_mask, div_mask) = mask_nodes

            if div_qk != div_mask:
                logger.debug("fuse_attention: skip since div_qk != div_mask")
                continue

            self.create_attention_node(gemm, gemm_qkv, layernorm_before_attention.output[0], reshape_qkv.output[0],
                                       attention_count == 0)
            nodes_to_remove.extend([reshape_qkv, transpose_qkv, matmul_qkv])
            nodes_to_remove.extend(qk_nodes)
            nodes_to_remove.extend(q_nodes)
            nodes_to_remove.extend(k_nodes)
            nodes_to_remove.extend(v_nodes)
            nodes_to_remove.extend(mask_nodes)
            attention_count += 1

        self.remove_nodes(nodes_to_remove)
        self.prune_graph()
        logger.info(f"Fused Attention count:{attention_count}")

    def create_attention_node(self, gemm, gemm_qkv, input, output, add_graph_input):
        attention_node_name = self.create_node_name('Attention')
        attention_node = onnx.helper.make_node('Attention',
                                               inputs=[input, gemm.input[1], gemm.input[2]],
                                               outputs=[attention_node_name + "_output"],
                                               name=attention_node_name)
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend(
            [onnx.helper.make_attribute("num_heads", self.num_heads),
             onnx.helper.make_attribute("unidirectional", 1)])

        matmul_node = onnx.helper.make_node('MatMul',
                                            inputs=[attention_node_name + "_output", gemm_qkv.input[1]],
                                            outputs=[attention_node_name + "_matmul_output"],
                                            name=attention_node_name + "_matmul")

        add_node = onnx.helper.make_node('Add',
                                         inputs=[attention_node_name + "_matmul_output", gemm_qkv.input[2]],
                                         outputs=[output],
                                         name=attention_node_name + "_add")

        self.add_node(attention_node)
        self.add_node(matmul_node)
        self.add_node(add_node)

    def postprocess(self):
        """
        Remove extra reshape nodes.
        """
        logger.debug(f"start postprocessing...")

        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        reshape_count = 0
        for gemm_node in self.get_nodes_by_op_type("Gemm"):
            reshape_after_gemm = self.find_first_child_by_type(gemm_node,
                                                               'Reshape',
                                                               input_name_to_nodes,
                                                               recursive=False)

            return_indice = []
            nodes = self.match_parent_path(gemm_node, ['Reshape', 'FastGelu'], [0, 0], output_name_to_node)
            if nodes is None:
                nodes = self.match_parent_path(gemm_node, ['Reshape', 'LayerNormalization'], [0, 0],
                                               output_name_to_node)
                if nodes is None:
                    continue
            (reshape_before_gemm, root_node) = nodes

            matmul_node_name = self.create_node_name('MatMul', 'FullyConnect_MatMul')
            matmul_node = onnx.helper.make_node('MatMul',
                                                inputs=[matmul_node_name + "_input", gemm_node.input[1]],
                                                outputs=[matmul_node_name + "_output"],
                                                name=matmul_node_name)

            add_node_name = self.create_node_name('Add', 'FullyConnect_Add')
            add_node = onnx.helper.make_node('Add',
                                             inputs=[matmul_node_name + "_output", gemm_node.input[2]],
                                             outputs=[add_node_name + "_output"],
                                             name=add_node_name)

            root_node.output[0] = matmul_node_name + "_input"
            self.replace_input_of_all_nodes(reshape_after_gemm.output[0], add_node_name + "_output")

            self.add_node(matmul_node)
            self.add_node(add_node)

            reshape_count += 2

        self.prune_graph()
        logger.info(f"Remove Reshape count:{reshape_count}")
