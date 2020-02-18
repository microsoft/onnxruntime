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
from BertOnnxModelTF import BertOnnxModelTF

logger = logging.getLogger(__name__)

class BertOnnxModelKeras(BertOnnxModelTF):
    def __init(self, model, num_heads, hidden_size, sequence_length, input_int32, float16, gpu_only):
        super().__init__(model, model, num_heads, hidden_size, sequence_length, input_int32, float16, gpu_only)

    def fuse_attention(self):
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        attention_count = 0

        skip_layer_norm_nodes = self.get_nodes_by_op_type("SkipLayerNormalization")
        for normalize_node in skip_layer_norm_nodes:
            # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
            parent = self.get_parent(normalize_node, 0)
            if parent is None or parent.op_type not in ["SkipLayerNormalization", "EmbedLayerNormalization"]:
                if parent.op_type == 'Add':
                    parent = self.get_parent(normalize_node, 1)
                    if parent is None or parent.op_type not in ["SkipLayerNormalization", "EmbedLayerNormalization"]:
                        logger.debug("First input for skiplayernorm: {}".format(parent.op_type if parent is not None else None))
                        continue
                else:
                    logger.debug("First input for skiplayernorm: {}".format(parent.op_type if parent is not None else None))
                    continue
            else:
                # TODO: shall we add back the checking of children op types.
                pass

            qkv_nodes = self.match_parent_path(normalize_node,
                                               ['Add', 'Reshape', 'MatMul', 'Reshape', 'Transpose', 'MatMul'],
                                               [ None,     0,         0,        0,         0,           0])
            if qkv_nodes is None:
                logger.debug("Failed to match qkv nodes")
                continue
            (add, extra_reshape_0, matmul, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
            logger.debug("Matched qkv nodes")

            v_nodes = self.match_parent_path(matmul_qkv,
                                             ['Transpose', 'Reshape', 'Add', 'Reshape', 'MatMul'],
                                             [ 1,           0,         0,     0,         0])
            if v_nodes is None:
                logger.debug("Failed to match v path")
                continue
            (transpose_v, reshape_v, add_v, extra_reshape_1, matmul_v) = v_nodes

            qk_nodes = self.match_parent_path(matmul_qkv, ['Softmax', 'Sub', 'MatMul'], [0, 0, 0])
            if qk_nodes is not None:
                (softmax_qk, sub_qk, matmul_qk) = qk_nodes
                q_nodes = self.match_parent_path(matmul_qk,
                                             ['Mul', 'Transpose', 'Reshape', 'Add', 'Reshape', 'MatMul'],
                                             [ 0,     None,           0,         0,     0,         0])
                if q_nodes is not None:                             
                    (mul_q, transpose_q, reshape_q, add_q, extra_reshape_2, matmul_q) = q_nodes
                                             
            else:
                qk_nodes = self.match_parent_path(matmul_qkv, ['Softmax', 'Add', 'Mul', 'MatMul'], [0, 0, 0, None])
                if qk_nodes is None:
                    logger.debug("Failed to match qk path")
                    continue
                (softmax_qk, add_qk, mul_qk, matmul_qk) = qk_nodes

                q_nodes = self.match_parent_path(matmul_qk,
                                                 ['Transpose', 'Reshape', 'Add', 'Reshape', 'MatMul'],
                                                 [ 0,           0,         0,     0,         0])
                if q_nodes is not None:
                    (transpose_q, reshape_q, add_q, extra_reshape_2, matmul_q) = q_nodes

            if q_nodes is None:
                logger.debug("Failed to match q path")
                continue

            k_nodes = self.match_parent_path(matmul_qk,
                                             ['Transpose', 'Reshape', 'Add', 'Reshape', 'MatMul'],
                                             [ 1,           0,         0,     0,         0])
            if k_nodes is None:
                logger.debug("Failed to match k path")
                continue
            (transpose_k, reshape_k, add_k, extra_reshape_3, matmul_k) = k_nodes


            mask_nodes = self.match_parent_path(qk_nodes[1],
                                                 ['Mul', 'Sub', 'Reshape', 'Cast'],
                                                 [ 1,     None,  1,         0])
            if mask_nodes is None:
                mask_nodes = self.match_parent_path(qk_nodes[1],
                                                 ['Mul', 'Sub', 'Cast', 'Slice', 'Unsqueeze'],
                                                 [ 1,     1,     1,      0,       0])
                if mask_nodes is None:                           
                    logger.debug("Failed to match mask path")
                    continue
                (mul_mask, sub_mask, cast_mask, slice_mask, unsqueeze_mask) = mask_nodes
            else:
                (mul_mask, sub_mask, reshape_mask, cast_mask) = mask_nodes

            if not self.has_constant_input(sub_mask, 1):
                logger.debug("Sub node expected to have an input with constant value 1.0.")
                continue

            root_input = matmul_v.input[0]
            root_node = output_name_to_node[root_input]
            is_same_root = (root_node == parent or (root_node.op_type == 'Reshape' and root_node.input[0] == parent.output[0]))
            if matmul_q.input[0] == root_input and matmul_v.input[0] == root_input and is_same_root:
                mask_index = self.process_mask(mask_nodes[-1].input[0])
                logger.debug("Create an Attention node.")
                self.create_attention_node(mask_index, matmul_q, matmul_k, matmul_v, add_q, add_k, add_v, parent.output[0], reshape_qkv.output[0])
                nodes_to_remove.extend([reshape_qkv, transpose_qkv, matmul_qkv])
                nodes_to_remove.extend(qk_nodes)
                nodes_to_remove.extend(q_nodes)
                nodes_to_remove.extend(k_nodes)
                nodes_to_remove.extend(v_nodes)
                nodes_to_remove.extend(mask_nodes)
                if root_node.op_type == 'Reshape':
                    nodes_to_remove.append(root_node)
                attention_count += 1
                
                nodes_to_remove.append(extra_reshape_0)
                self.replace_node_input(add, extra_reshape_0.output[0], matmul.output[0])
            else:
                logger.debug("Root node not matched.")
                continue
        self.remove_nodes(nodes_to_remove)
        self.update_graph()
        logger.info(f"Fused Attention count:{attention_count}")

    def fuse_embedding(self, node, output_name_to_node):
        assert node.op_type == 'LayerNormalization'
        pos_embed_path2 = self.match_parent_path(
            node,
            ['Add', 'Add', 'Gather'],
            [ 0,     0,     0],
            output_name_to_node)
        if pos_embed_path2 is None:
            logger.debug("failed to match pos_embed_path")
            return False

        skip_node, add_node, gather_node = pos_embed_path2
        pos_initializer = self.get_initializer(add_node.input[1])
        if pos_initializer is None:
            return False
            
        temp = numpy_helper.to_array(pos_initializer)
        if len(temp.shape) == 3 and temp.shape[0] == 1:
            tensor = numpy_helper.from_array(temp.reshape((temp.shape[1],temp.shape[2])), "position_embedding")
            self.add_initializer(tensor)
            logger.info("Found position embedding. name:{}, shape:{}".format(pos_initializer.name, temp.shape[1:]))
            position_embedding = "position_embedding"
        else:
            logger.info("Failed to find position embedding. name:{}, shape:{}".format(pos_initializer.name, temp.shape))
            return False


        word_initializer = self.get_initializer(gather_node.input[0])
        if word_initializer is None:
            return False
            
        temp = numpy_helper.to_array(word_initializer)
        if len(temp.shape) == 2:
            logger.info("Found word embedding. name:{}, shape:{}".format(word_initializer.name, temp.shape))
            word_embedding = word_initializer.name
        else:
            logger.info("Failed to find word embedding. name:{}, shape:{}".format(word_initializer.name, temp.shape))
            return False

        gather = self.get_parent(skip_node, 1, output_name_to_node)
        if gather is None or gather.op_type != "Gather":
            return False

        segment_initializer = self.get_initializer(gather.input[0])
        if segment_initializer is None:
            return False
            
        temp = numpy_helper.to_array(segment_initializer)
        if len(temp.shape) == 2:
            logger.info("Found segment embedding. name:{}, shape:{}".format(segment_initializer.name, temp.shape))
            segment_embedding = segment_initializer.name
        else:
            logger.info("Failed to find segment embedding. name:{}, shape:{}".format(segment_initializer.name, temp.shape))
            return False
        
        logger.info("Create Embedding node")
        self.create_embedding_subgraph(node, word_embedding, segment_embedding, position_embedding)
        return True

    def process_embedding(self):
        """
        Automatically detect word, segment and position embeddings.
        """
        logger.info("start processing embedding layer...")
        output_name_to_node = self.output_name_to_node()
        for node in self.nodes():
            if node.op_type == 'LayerNormalization':
                if self.fuse_embedding(node, output_name_to_node):
                    return

    def remove_extra_reshape(self):
        skiplayernorm_nodes = self.get_nodes_by_op_type("SkipLayerNormalization")
        reshape_removed = 0
        for skiplayernorm_node in skiplayernorm_nodes:
            path = self.match_parent_path(
                        skiplayernorm_node,
                        ['Add', 'Reshape', 'MatMul', 'Reshape', 'Gelu', 'Add', 'Reshape', 'MatMul', 'SkipLayerNormalization'],
                        [ 0,     0,         0,        0,         0,      0,     0,         0,        0])
            if path is None:
                continue

            add_1, reshape_1, matmul_1, reshape_2, gelu, add_2, reshape_3, matmul_2, skiplayernorm = path
            add_2.input[0] = matmul_2.output[0]
            self.remove_node(reshape_3)
            matmul_1.input[0] = gelu.output[0]
            self.remove_node(reshape_2)
            add_1.input[0] = matmul_1.output[0]
            self.remove_node(reshape_1)
            reshape_removed += 3

        logger.info(f"Remove {reshape_removed} Reshape nodes.")

    def remove_extra_reshape_2(self):
        skiplayernorm_nodes = self.get_nodes_by_op_type("SkipLayerNormalization")
        reshape_removed = 0
        for skiplayernorm_node in skiplayernorm_nodes:
            path = self.match_parent_path(
                        skiplayernorm_node,
                        ['Add', 'Reshape', 'MatMul', 'Reshape', 'Gelu', 'Add', 'Reshape', 'MatMul', 'Reshape', 'SkipLayerNormalization'],
                        [ None,     0,         0,        0,         0,      0,     0,         0,        0,         0])
            if path is None:
                continue

            add_1, reshape_1, matmul_1, reshape_2, gelu, add_2, reshape_3, matmul_2, reshape_4, skiplayernorm = path
            
            matmul_2.input[0] = skiplayernorm.output[0]
            self.remove_node(reshape_4)
            
            add_2.input[0] = matmul_2.output[0]
            self.remove_node(reshape_3)
            
            matmul_1.input[0] = gelu.output[0]
            self.remove_node(reshape_2)
            
            add_1.input[0] = matmul_1.output[0]
            self.remove_node(reshape_1)
            
            reshape_removed += 4

        logger.info(f"Remove {reshape_removed} Reshape nodes.")

    def postprocess(self):
        self.remove_extra_reshape()
        self.remove_extra_reshape_2()
        self.prune_graph()
