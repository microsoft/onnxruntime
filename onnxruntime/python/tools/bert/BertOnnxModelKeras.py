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

    def match_mask_path(self, add_or_sub_before_softmax):
        mask_nodes = self.match_parent_path(add_or_sub_before_softmax, ['Mul', 'Sub', 'Reshape', 'Cast'],
                                            [1, None, 1, 0])
        if mask_nodes is not None:
            return mask_nodes

        mask_nodes = self.match_parent_path(add_or_sub_before_softmax, ['Mul', 'Sub', 'Cast', 'Slice', 'Unsqueeze'],
                                            [1, 1, 1, 0, 0])
        if mask_nodes is not None:
            return mask_nodes

        mask_nodes = self.match_parent_path(add_or_sub_before_softmax, ['Mul', 'Sub', 'Cast', 'Unsqueeze', 'Unsqueeze'],
                                            [1, None, 1, 0, 0])
        return mask_nodes

    def check_attention_input(self, matmul_q, matmul_k, matmul_v, parent, output_name_to_node):
        reshape_nodes = []

        for x in [matmul_q, matmul_k, matmul_v]:
            root_input = x.input[0]
            root_node = output_name_to_node[root_input]
            if root_node == parent:
                continue
            if root_node.op_type == 'Reshape' and root_node.input[0] == parent.output[0]:
                reshape_nodes.append(root_node)
                continue
            logger.debug(f"Check attention input failed:{root_input}, {parent.output[0]}")
            return False, []

        return True, reshape_nodes

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
                        logger.debug(
                            "First input for skiplayernorm: {}".format(parent.op_type if parent is not None else None))
                        continue
                else:
                    logger.debug(
                        "First input for skiplayernorm: {}".format(parent.op_type if parent is not None else None))
                    continue
            else:
                # TODO: shall we add back the checking of children op types.
                pass

            qkv_nodes = self.match_parent_path(normalize_node,
                                               ['Add', 'Reshape', 'MatMul', 'Reshape', 'Transpose', 'MatMul'],
                                               [None, 0, 0, 0, 0, 0])
            if qkv_nodes is None:
                logger.debug("Failed to match qkv nodes")
                continue
            (add, extra_reshape_0, matmul, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
            logger.debug("Matched qkv nodes")

            v_nodes = self.match_parent_path(matmul_qkv, ['Transpose', 'Reshape', 'Add', 'Reshape', 'MatMul'],
                                             [1, 0, 0, 0, 0])
            if v_nodes is None:
                logger.debug("Failed to match v path")
                continue
            (transpose_v, reshape_v, add_v, extra_reshape_1, matmul_v) = v_nodes

            qk_nodes = self.match_parent_path(matmul_qkv, ['Softmax', 'Sub', 'MatMul'], [0, 0, 0])
            if qk_nodes is not None:
                (softmax_qk, sub_qk, matmul_qk) = qk_nodes
                q_nodes = self.match_parent_path(matmul_qk, ['Mul', 'Transpose', 'Reshape', 'Add', 'Reshape', 'MatMul'],
                                                 [0, None, 0, 0, 0, 0])
                if q_nodes is not None:
                    (mul_q, transpose_q, reshape_q, add_q, extra_reshape_2, matmul_q) = q_nodes

            else:
                qk_nodes = self.match_parent_path(matmul_qkv, ['Softmax', 'Add', 'Mul', 'MatMul'], [0, 0, 0, None])
                if qk_nodes is None:
                    qk_nodes = self.match_parent_path(matmul_qkv, ['Softmax', 'Add', 'Div', 'MatMul'], [0, 0, 0, None])
                    if qk_nodes is None:
                        logger.debug("Failed to match qk path")
                        continue
                (softmax_qk, add_qk, mul_qk, matmul_qk) = qk_nodes

                q_nodes = self.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'Reshape', 'MatMul'],
                                                 [0, 0, 0, 0, 0])
                if q_nodes is not None:
                    (transpose_q, reshape_q, add_q, extra_reshape_2, matmul_q) = q_nodes

            if q_nodes is None:
                logger.debug("Failed to match q path")
                continue

            k_nodes = self.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'Reshape', 'MatMul'],
                                             [1, 0, 0, 0, 0])
            if k_nodes is None:
                logger.debug("Failed to match k path")
                continue
            (transpose_k, reshape_k, add_k, extra_reshape_3, matmul_k) = k_nodes

            mask_nodes = self.match_mask_path(qk_nodes[1])
            if mask_nodes is None:
                logger.debug("Failed to match mask path")
                continue
            if not self.has_constant_input(mask_nodes[1], 1):
                logger.debug("Sub node expected to have an input with constant value 1.0.")
                continue

            is_same_root, reshape_nodes = self.check_attention_input(matmul_q, matmul_k, matmul_v, parent,
                                                                     output_name_to_node)
            if is_same_root:
                mask_index = self.process_mask(mask_nodes[-1].input[0])
                logger.debug("Create an Attention node.")
                self.create_attention_node(mask_index, matmul_q, matmul_k, matmul_v, add_q, add_k, add_v,
                                           parent.output[0], reshape_qkv.output[0])
                nodes_to_remove.extend([reshape_qkv, transpose_qkv, matmul_qkv])
                nodes_to_remove.extend(qk_nodes)
                nodes_to_remove.extend(q_nodes)
                nodes_to_remove.extend(k_nodes)
                nodes_to_remove.extend(v_nodes)
                nodes_to_remove.extend(mask_nodes)
                nodes_to_remove.extend(reshape_nodes)
                attention_count += 1
                nodes_to_remove.append(extra_reshape_0)
                self.replace_node_input(add, extra_reshape_0.output[0], matmul.output[0])
            else:
                logger.debug("Root node not matched.")
                continue
        self.remove_nodes(nodes_to_remove)
        self.update_graph()
        logger.info(f"Fused Attention count:{attention_count}")

    def preprocess(self):
        self.process_embedding()
        self.fuse_mask()
        self.skip_reshape()

    def skip_reshape(self):
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        attention_count = 0

        count = 0
        reshape_nodes = self.get_nodes_by_op_type("Reshape")
        for reshape_node in reshape_nodes:
            parent = self.get_parent(reshape_node, 0)
            if parent is not None and parent.op_type == "Reshape":
                reshape_node.input[0] = parent.input[0]
                count += 1

        if count > 0:
            logger.info(f"Skip consequent Reshape count: {count}")

    def fuse_embedding(self, node, output_name_to_node):
        assert node.op_type == 'LayerNormalization'
        logger.debug(f"start fusing embedding from node with output={node.output[0]}...")
        word_embed_path = self.match_parent_path(node, ['Add', 'Add', 'Gather'], [0, 0, 0], output_name_to_node)
        if word_embed_path is None:
            logger.debug("failed to match word_embed_path")
            return False

        skip_node, add_node, gather_node = word_embed_path

        word_initializer = self.get_initializer(gather_node.input[0])
        if word_initializer is None:
            logger.debug("failed to get word initializer")
            return False

        temp = numpy_helper.to_array(word_initializer)
        if len(temp.shape) == 2:
            logger.info("Found word embedding. name:{}, shape:{}".format(word_initializer.name, temp.shape))
            word_embedding = word_initializer.name
        else:
            logger.info("Failed to find word embedding. name:{}, shape:{}".format(word_initializer.name, temp.shape))
            return False

        pos_initializer = self.get_initializer(add_node.input[1])
        if pos_initializer is not None:
            temp = numpy_helper.to_array(pos_initializer)
            if len(temp.shape) == 3 and temp.shape[0] == 1:
                tensor = numpy_helper.from_array(temp.reshape((temp.shape[1], temp.shape[2])), "position_embedding")
                self.add_initializer(tensor)
                logger.info("Found position embedding. name:{}, shape:{}".format(pos_initializer.name, temp.shape[1:]))
                position_embedding = "position_embedding"
            else:
                logger.info("Failed to find position embedding. name:{}, shape:{}".format(
                    pos_initializer.name, temp.shape))
                return False
        else:
            pos_embed_path = self.match_parent_path(add_node, ['Gather', 'Slice'], [1, 1], output_name_to_node)
            if pos_embed_path is None:
                logger.debug("failed to match pos_embed_path")
                return False

            pos_gather, pos_slice = pos_embed_path
            pos_initializer = self.get_initializer(pos_gather.input[0])
            if pos_initializer is None:
                logger.debug("failed to get pos initializer")
                return False

            temp = numpy_helper.to_array(pos_initializer)
            if len(temp.shape) == 2:
                logger.info("Found word embedding. name:{}, shape:{}".format(pos_initializer.name, temp.shape))
                position_embedding = pos_initializer.name
            else:
                logger.info("Failed to find position embedding. name:{}, shape:{}".format(
                    pos_initializer.name, temp.shape))
                return False

        gather = self.get_parent(skip_node, 1, output_name_to_node)
        if gather is None or gather.op_type != "Gather":
            logger.debug("failed to get gather")
            return False

        segment_initializer = self.get_initializer(gather.input[0])
        if segment_initializer is None:
            logger.debug("failed to get segment initializer")
            return False

        temp = numpy_helper.to_array(segment_initializer)
        if len(temp.shape) == 2:
            logger.info("Found segment embedding. name:{}, shape:{}".format(segment_initializer.name, temp.shape))
            segment_embedding = segment_initializer.name
        else:
            logger.info("Failed to find segment embedding. name:{}, shape:{}".format(
                segment_initializer.name, temp.shape))
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
                break

    def fuse_mask(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == 'Mul' and self.has_constant_input(node, -10000):
                mask_path = self.match_parent_path(node, ['Sub', 'Cast', 'Slice', 'Unsqueeze'], [0, 1, 0, 0])
                if mask_path is None:
                    continue
                sub_node, cast_node, slice_node, unsqueeze_node = mask_path

                mask_input_name = next(iter(self.mask_indice))
                if unsqueeze_node.input[0] != mask_input_name:
                    print("Cast input {} is not mask input{}".format(unsqueeze_node.input[0], mask_input_name))
                    continue

                unsqueeze_added_1 = onnx.helper.make_node('Unsqueeze',
                                                          inputs=[mask_input_name],
                                                          outputs=['mask_fuse_unsqueeze1_output'],
                                                          name='Mask_UnSqueeze_1',
                                                          axes=[1])

                unsqueeze_added_2 = onnx.helper.make_node('Unsqueeze',
                                                          inputs=['mask_fuse_unsqueeze1_output'],
                                                          outputs=['mask_fuse_unsqueeze2_output'],
                                                          name='Mask_UnSqueeze_2',
                                                          axes=[2])

                #self.replace_node_input(cast_node, cast_node.input[0], 'mask_fuse_unsqueeze2_output')
                cast_node_2 = onnx.helper.make_node('Cast',
                                                    inputs=['mask_fuse_unsqueeze2_output'],
                                                    outputs=['mask_fuse_cast_output'])
                cast_node_2.attribute.extend([onnx.helper.make_attribute("to", 1)])
                self.replace_node_input(sub_node, sub_node.input[1], 'mask_fuse_cast_output')

                nodes_to_remove.extend([slice_node, unsqueeze_node, cast_node])
                self.add_node(unsqueeze_added_1)
                self.add_node(unsqueeze_added_2)
                self.add_node(cast_node_2)

        self.remove_nodes(nodes_to_remove)

        # Prune graph is done after removing nodes to remove island nodes.
        if len(nodes_to_remove) > 0:
            self.prune_graph()

        logger.info("Fused mask" if len(nodes_to_remove) > 0 else "Failed to fuse mask")

    def remove_extra_reshape(self):
        skiplayernorm_nodes = self.get_nodes_by_op_type("SkipLayerNormalization")
        reshape_removed = 0
        for skiplayernorm_node in skiplayernorm_nodes:
            path = self.match_parent_path(
                skiplayernorm_node,
                ['Add', 'Reshape', 'MatMul', 'Reshape', 'Gelu', 'Add', 'Reshape', 'MatMul', 'SkipLayerNormalization'],
                [0, 0, 0, 0, 0, 0, 0, 0, 0])
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

        return reshape_removed

    def remove_extra_reshape_2(self):
        skiplayernorm_nodes = self.get_nodes_by_op_type("SkipLayerNormalization")
        reshape_removed = 0
        for skiplayernorm_node in skiplayernorm_nodes:
            path = self.match_parent_path(
                skiplayernorm_node,
                ['Add', 'Reshape', 'MatMul', 'Reshape', 'Gelu', 'Add', 'Reshape', 'MatMul', 'Reshape', 'SkipLayerNormalization'],
                [None, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # yapf: disable
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

        return reshape_removed

    def postprocess(self):
        reshape_removed = self.remove_extra_reshape() + self.remove_extra_reshape_2()
        logger.info(f"Remove {reshape_removed} Reshape nodes.")

        self.prune_graph()

    """
     Fuse Gelu with Erf into one node:
                   +------------------------------------------+
                   |                                          |
                   |                                          v
                [root] --> Div -----> Erf  --> Add --> Mul -->Mul
                          (B=1.4142...)       (A=1)   (A=0.5)

     Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
    """

    def fuse_gelu_with_elf(self):
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        nodes_to_add = []

        for node in self.get_nodes_by_op_type('Erf'):
            erf_node = node

            if erf_node.output[0] not in input_name_to_nodes:
                continue
            children = input_name_to_nodes[erf_node.output[0]]
            if len(children) != 1 or children[0].op_type != 'Add':
                continue
            add_after_erf = children[0]

            if not self.has_constant_input(add_after_erf, 1):
                continue

            if add_after_erf.output[0] not in input_name_to_nodes:
                continue
            children = input_name_to_nodes[add_after_erf.output[0]]
            if len(children) != 1 or children[0].op_type != 'Mul':
                continue
            mul_after_erf = children[0]

            if not self.has_constant_input(mul_after_erf, 0.5):
                continue

            if mul_after_erf.output[0] not in input_name_to_nodes:
                continue
            children = input_name_to_nodes[mul_after_erf.output[0]]
            if len(children) != 1 or children[0].op_type != 'Mul':
                continue
            mul = children[0]

            div = self.match_parent(erf_node, 'Div', 0, output_name_to_node)
            if div is None:
                continue

            sqrt_node = None
            if self.find_constant_input(div, 1.4142, delta=0.001) != 1:
                sqrt_node = self.match_parent(div, 'Sqrt', 1, output_name_to_node)
                if sqrt_node is None:
                    continue
                if not self.has_constant_input(sqrt_node, 2.0):
                    continue

            root_node = self.get_parent(div, 0, output_name_to_node)
            if root_node is None:
                continue

            if root_node.output[0] not in mul.input:
                continue

            subgraph_nodes = [div, erf_node, add_after_erf, mul_after_erf, mul]
            if sqrt_node:
                subgraph_nodes.append(sqrt_node)

            if not self.is_safe_to_fuse_nodes(subgraph_nodes, [mul.output[0]], input_name_to_nodes,
                                              output_name_to_node):
                continue

            nodes_to_remove.extend(subgraph_nodes)
            gelu_node = onnx.helper.make_node('Gelu', inputs=[root_node.output[0]], outputs=[mul.output[0]])
            gelu_node.domain = "com.microsoft"
            nodes_to_add.append(gelu_node)

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)
        if len(nodes_to_add) > 0:
            logger.info(f"Fused Gelu count:{len(nodes_to_add)}")
        else:
            super().fuse_gelu_with_elf()
