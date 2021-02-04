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
from onnx import ModelProto, TensorProto, numpy_helper, helper
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)


class BertOnnxModelTF(BertOnnxModel):
    def __init(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)

    def remove_identity(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == 'Identity':
                if not self.find_graph_output(node.output[0]):
                    self.replace_input_of_all_nodes(node.output[0], node.input[0])
                    nodes_to_remove.append(node)
        self.remove_nodes(nodes_to_remove)
        logger.info(f"Removed Identity count: {len(nodes_to_remove)}")

    def fuse_mask(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == 'Sub':
                parent_path_constant = self.match_parent_path(
                    node,
                    ['Reshape', 'Mul', 'ConstantOfShape', 'Cast', 'Concat', 'Unsqueeze', 'Cast', 'Squeeze', 'Slice', 'Cast', 'Shape'],
                    [        1,     0,                 0,      0,        0,           0,      0,         0,        0,     0,       0]) # yapf: disable
                if parent_path_constant is None:
                    continue
                reshape_node_0, mul_node_0, constantofshape_node, cast_node_0, concat_node_0, unsqueeze_node, cast_node_1, squeeze_node, slice_node, cast_node_2, shape_node = parent_path_constant

                parent_path_mask = self.match_parent_path(
                    mul_node_0,
                    ['Cast', 'Reshape', 'Cast', 'Concat', 'Unsqueeze'],
                    [     1,     0,          1,       0,            0]) # yapf: disable

                if parent_path_mask is None:
                    continue

                cast_node_3, reshape_node_1, cast_node_4, concat_node_1, unsqueeze_node_1 = parent_path_mask

                if not unsqueeze_node_1 == unsqueeze_node:
                    continue

                unsqueeze_added_1 = onnx.helper.make_node('Unsqueeze',
                                                          inputs=[reshape_node_1.input[0]],
                                                          outputs=['mask_fuse_unsqueeze1_output'],
                                                          name='Mask_UnSqueeze_1',
                                                          axes=[1])

                unsqueeze_added_2 = onnx.helper.make_node('Unsqueeze',
                                                          inputs=['mask_fuse_unsqueeze1_output'],
                                                          outputs=[cast_node_3.input[0]],
                                                          name='Mask_UnSqueeze_2',
                                                          axes=[2])
                node.input[1] = cast_node_3.output[0]

                nodes_to_remove.extend([
                    reshape_node_0, mul_node_0, constantofshape_node, cast_node_0, concat_node_0, unsqueeze_node,
                    cast_node_1, squeeze_node, slice_node, cast_node_2, shape_node
                ])
                nodes_to_remove.extend([reshape_node_1, cast_node_4, concat_node_1])
                self.add_node(unsqueeze_added_1)
                self.add_node(unsqueeze_added_2)

        self.remove_nodes(nodes_to_remove)
        if len(nodes_to_remove) > 0:
            logger.info("Fused mask")
        else:
            self.fuse_mask_2()

    def fuse_mask_2(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == 'Mul' and self.has_constant_input(node, -10000):
                mask_path = self.match_parent_path(node, ['Sub', 'Unsqueeze', 'Mul', 'Cast', 'Reshape', 'Cast'],
                                                   [0, 1, 0, 1, 0, 0])
                if mask_path is None:
                    continue
                sub_node, unsqueeze_node, mul_node, cast_node_0, reshape_node_0, cast_node_1 = mask_path

                mask_input_name = self.attention_mask.get_first_mask()

                if cast_node_1.input[0] != mask_input_name:
                    print("Cast input {} is not mask input{}".format(cast_node_1.input[0], mask_input_name))
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

                cast_node_2 = onnx.helper.make_node('Cast',
                                                    inputs=['mask_fuse_unsqueeze2_output'],
                                                    outputs=['mask_fuse_cast_output'])
                cast_node_2.attribute.extend([onnx.helper.make_attribute("to", 1)])
                self.replace_node_input(sub_node, sub_node.input[1], 'mask_fuse_cast_output')

                nodes_to_remove.extend([unsqueeze_node, mul_node, cast_node_0, reshape_node_0, cast_node_1])
                self.add_node(unsqueeze_added_1)
                self.add_node(unsqueeze_added_2)
                self.add_node(cast_node_2)

        self.remove_nodes(nodes_to_remove)

        # Prune graph is done after removing nodes to remove island nodes.
        if len(nodes_to_remove) > 0:
            self.prune_graph()

        logger.info("Fused mask" if len(nodes_to_remove) > 0 else "Failed to fuse mask")

    def get_2d_initializers_from_parent_subgraphs(self, current_node):
        """
        Find initializers that is 2D. Returns a dictionary with name as key and shape as value.
        """
        parent_nodes = self.get_parent_subgraph_nodes(current_node, [])
        initializers = {}
        for node in parent_nodes:
            for input in node.input:
                initializer = self.get_initializer(input)
                if initializer:
                    temp = numpy_helper.to_array(initializer)
                    if len(temp.shape) == 2:
                        initializers[initializer.name] = temp.shape

        return initializers

    def find_segment_ids(self, segment_embedding, input_ids):
        input_name_to_nodes = self.input_name_to_nodes()
        if segment_embedding not in input_name_to_nodes:
            return None

        nodes = input_name_to_nodes[segment_embedding]
        if len(nodes) != 1:
            return None

        graph_inputs = self.get_graph_inputs(nodes[0], recursive=True)
        if len(graph_inputs) > 1:
            print("Found multiple candidates of segment_ids", graph_inputs)
            return None
        # Find segment ids in graph inputs. The segment id input must not be the same as input_ids.
        if len(graph_inputs) == 1 and graph_inputs[0] != input_ids:
            return graph_inputs[0]

        # If the segment id candidate is the same as the input_ids, try to assign alternative segment ids and simplify the graph if needed.
        segment_ids = nodes[0].input[1]
        _, segment_id_path, _ = self.match_parent_paths(
            nodes[0], [(["ConstantOfShape", "Cast", "Concat", "Slice", "Cast", "Shape"], [1, 0, 0, 0, 0, 0]),
                       (["ConstantOfShape", "Cast", "Concat", "Unsqueeze", "Squeeze", "Slice", "Cast", "Shape"
                         ], [1, 0, 0, 0, 0, 0, 0, 0])], None)

        if segment_id_path and input_ids and input_ids == segment_id_path[-1].input[0]:
            logger.debug("Simplify semgent id path...")
            self.add_node(helper.make_node('Shape', inputs=[input_ids], outputs=["input_shape"]))
            constantofshape_node = segment_id_path[0]
            constantofshape_value = helper.get_attribute_value(constantofshape_node.attribute[0])
            self.add_node(
                helper.make_node('ConstantOfShape',
                                 inputs=["input_shape"],
                                 outputs=["zeros_for_input_shape"],
                                 value=constantofshape_value))
            segment_ids = "zeros_for_input_shape"
        return segment_ids

    def find_input_ids(self, word_embedding):
        input_name_to_nodes = self.input_name_to_nodes()
        if word_embedding not in input_name_to_nodes:
            return None

        nodes = input_name_to_nodes[word_embedding]
        if len(nodes) != 1:
            return None

        graph_inputs = self.get_graph_inputs(nodes[0], recursive=True)
        if len(graph_inputs) == 1:
            return graph_inputs[0]

        print("Found multiple candidates of input_ids", graph_inputs)
        return None

    def find_mask_input(self, excluded_graph_inputs):
        for node in self.nodes():
            if node.op_type == 'Softmax':
                mask_path = self.match_parent_path(node, ['Add', 'Mul', 'Sub', 'Cast', 'Slice', 'Unsqueeze'],
                                                   [0, 1, None, 1, 0, 0])
                if mask_path is None:
                    continue
                add_node, mul_node, sub_node, cast_node, slice_node, unsqueeze_node = mask_path
                if self.has_constant_input(mul_node, -10000) and self.has_constant_input(sub_node, 1):
                    graph_inputs = self.get_graph_inputs(sub_node, recursive=True)
                    inputs = [input for input in graph_inputs if input not in excluded_graph_inputs]
                    if len(inputs) > 1:
                        print("Found multiple candidates of mask input", inputs)
                        return None
                    if len(inputs) == 1:
                        return inputs[0]
                    # Duplicated input found. Try to simplify the graph.
                    path_to_be_simplified = self.match_parent_path(
                        mask_path[-1],
                        ["ConstantOfShape", "Cast", "Concat", "Unsqueeze", "Squeeze", "Slice", "Cast", "Shape"],
                        [0, 0, 0, 0, 0, 0, 0, 0])
                    duplicated_inputs = [input for input in graph_inputs if input in excluded_graph_inputs]
                    # Simplify graph for dynamic axes.
                    if path_to_be_simplified and duplicated_inputs and len(
                            duplicated_inputs) == 1 and duplicated_inputs[0] == path_to_be_simplified[-1].input[0]:
                        logger.debug("Simplify semgent id path...")
                        constantofshape_node = path_to_be_simplified[0]
                        constantofshape_value = helper.get_attribute_value(constantofshape_node.attribute[0])
                        self.add_node(
                            helper.make_node('Shape', inputs=[duplicated_inputs[0]], outputs=["input_shape_for_mask"]))
                        self.add_node(
                            helper.make_node('ConstantOfShape',
                                             inputs=["input_shape_for_mask"],
                                             outputs=[unsqueeze_node.input[0]],
                                             value=constantofshape_value))
                    return unsqueeze_node.input[0]
        return None

    def create_embedding_subgraph(self, normalize_node, word_embedding, segment_embedding, position_embedding):
        input_ids = self.find_input_ids(word_embedding)
        if input_ids is None:
            logger.info("Failed to find input_ids. Cannot fuse embedding layer.")
            return False

        segment_ids = self.find_segment_ids(segment_embedding, input_ids)
        if segment_ids is None:
            logger.info("Failed to find segment_ids. Cannot fuse embedding layer.")
            return False

        mask_input = self.find_mask_input([segment_ids, input_ids])
        if mask_input is None:
            logger.info("Failed to find input_mask. Cannot fuse embedding layer.")
            return False

        self.bert_inputs = [input_ids, segment_ids, mask_input]

        mask_index = self.create_node_name('mask_index')
        self.attention_mask.set_mask_indice(mask_input, mask_index)

        if self.find_graph_input(input_ids).type.tensor_type.elem_type != TensorProto.INT32:
            casted, input_ids = self.utils.cast_graph_input_to_int32(input_ids)

        if self.find_graph_input(segment_ids):
            casted, segment_ids = self.utils.cast_graph_input_to_int32(segment_ids)
        else:
            segment_ids, segment_id_cast_node = self.utils.cast_input_to_int32(segment_ids)

        if self.find_graph_input(mask_input):
            casted, mask_input = self.utils.cast_graph_input_to_int32(mask_input)
        else:
            mask_input, mask_input_cast_node = self.utils.cast_input_to_int32(mask_input)

        embed_output = self.create_node_name('embed_output')
        embed_node = onnx.helper.make_node(
            'EmbedLayerNormalization',
            inputs=[
                input_ids,
                segment_ids,
                word_embedding,
                position_embedding,
                segment_embedding,
                normalize_node.input[1],  # gamma
                normalize_node.input[2],  # beta
                mask_input
            ],
            outputs=[embed_output, mask_index],
            name="EmbedLayer")
        embed_node.domain = "com.microsoft"
        self.replace_input_of_all_nodes(normalize_node.output[0], embed_output)
        self.add_node(embed_node)

    def process_embedding(self):
        """
        Automatically detect word, segment and position embeddings.
        """
        logger.info("start processing embedding layer...")
        output_name_to_node = self.output_name_to_node()

        layer_norm_nodes = self.get_nodes_by_op_type("LayerNormalization")
        for layer_norm_node in layer_norm_nodes:
            pos_embed_path = self.match_parent_path(layer_norm_node, ['Add', 'Reshape', 'Slice'], [0, 1, 0],
                                                    output_name_to_node)
            if pos_embed_path is None:
                continue

            add_node, reshape_node, slice_node = pos_embed_path
            initializer = self.get_initializer(slice_node.input[0])
            if initializer is None:
                continue

            temp = numpy_helper.to_array(initializer)
            if len(temp.shape) == 2:
                logger.info("Found position embedding. name:{}, shape:{}".format(initializer.name, temp.shape))
                position_embedding = initializer.name
            else:
                logger.info("Failed to find position embedding. name:{}, shape:{}".format(initializer.name, temp.shape))
                return

            first_parent = self.get_parent(add_node, 0, output_name_to_node)
            if first_parent is not None and first_parent.op_type == "Add":
                embeddings = self.get_2d_initializers_from_parent_subgraphs(first_parent)
                if len(embeddings) != 2:
                    logger.warning(
                        "Failed to find two embeddings (word and segment) from Add node. Found {}".format(embeddings))
                    return

                word_embedding = None
                segment_embedding = None
                for name, shape in embeddings.items():
                    if shape[0] == 2:
                        segment_embedding = name
                        logger.info("Found segment embedding. name:{}, shape:{}".format(name, shape))
                    else:
                        word_embedding = name
                        logger.info("Found words embedding. name:{}, shape:{}".format(name, shape))

                if word_embedding is None or segment_embedding is None:
                    logger.info("Failed to find both word and segment embedding")
                    return

                logger.info("Create Embedding node")
                self.create_embedding_subgraph(layer_norm_node, word_embedding, segment_embedding, position_embedding)
                # Prune graph to remove those original embedding nodes.
                self.prune_graph()
                break

    def check_attention_input(self, matmul_q, matmul_k, matmul_v, parent, output_name_to_node):
        for x in [matmul_q, matmul_k, matmul_v]:
            root_input = x.input[0]
            root_node = output_name_to_node[root_input]
            if root_node == parent:
                continue
            logger.debug(f"Check attention input failed:{root_input}, {parent.output[0]}")
            return False

        return True

    def fuse_attention(self):
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        attention_count = 0

        skip_layer_norm_nodes = self.get_nodes_by_op_type("SkipLayerNormalization")
        for normalize_node in skip_layer_norm_nodes:
            # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
            parent = self.get_parent(normalize_node, 1)
            if parent is None or parent.op_type not in ["SkipLayerNormalization", "LayerNormalization", "Reshape"]:
                parent = self.get_parent(normalize_node, 0)
                if parent is None or parent.op_type not in ["SkipLayerNormalization", "LayerNormalization", "Reshape"]:
                    logger.debug("Failed to match parent of normalize_node")
                    continue

            qkv_nodes = self.match_parent_path(normalize_node, ['Add', 'MatMul', 'Reshape', 'Transpose', 'MatMul'],
                                               [0, 0, 0, 0, 0])
            if qkv_nodes is None:
                qkv_nodes = self.match_parent_path(normalize_node, ['MatMul', 'Reshape', 'Transpose', 'MatMul'],
                                                   [1, 0, 0, 0])
                if qkv_nodes is None:
                    logger.debug("Failed to match qkv nodes")
                    continue

            (reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes[-3:]
            v_nodes = self.match_parent_path(matmul_qkv, ['Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0])
            if v_nodes is None:
                logger.debug("Failed to match v path")
                continue

            (transpose_v, reshape_v, add_v, matmul_v) = v_nodes
            qk_nodes = self.match_parent_path(matmul_qkv, ['Softmax', 'Add', "Mul", 'MatMul'], [0, 0, 0, 0])
            if qk_nodes is None:
                logger.debug("Failed to match qk_paths")
                continue
            (softmax_qk, add_qk, mul_qk, matmul_qk) = qk_nodes

            q_nodes = self.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'MatMul'], [0, 0, 0, 0])
            if q_nodes is None:
                logger.debug("Failed to match q path")
                continue
            (transpose_q, reshape_q, add_q, matmul_q) = q_nodes

            k_nodes = self.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0])
            if k_nodes is None:
                logger.debug("Failed to match k path")
                continue
            (transpose_k, reshape_k, add_k, matmul_k) = k_nodes

            mask_nodes = self.match_parent_path(add_qk, ['Mul', 'Sub', 'Unsqueeze'], [1, 0, 1])
            if mask_nodes is None:
                mask_nodes = self.match_parent_path(add_qk, ['Mul', 'Sub', 'Cast', 'Unsqueeze', 'Mul'], [1, 0, 1, 0, 0])
                if mask_nodes is None:
                    logger.debug("Failed to match mask path")
                    continue

            if not self.has_constant_input(mask_nodes[1], 1):
                logger.debug("Sub node expected to have an input with constant value 1.0.")
                continue

            # add a squeeze node to convert a 3-d mask to 2-d
            squeeze_node = self.match_parent_path(mask_nodes[-1], ['Squeeze'], [0])
            squeeze_node_name = "Squeeze_3d_to_2d_mask"
            squeeze_output_name = squeeze_node_name + "_output"
            if squeeze_node is None and len(mask_nodes) == 5:
                mask_input = mask_nodes[-1].input[1]
                self.add_node(
                    helper.make_node("Squeeze", [mask_input], [squeeze_output_name], squeeze_node_name, axes=[1]))
                mask_nodes[-1].input[0] = squeeze_output_name

            is_same_root = self.check_attention_input(matmul_q, matmul_k, matmul_v, parent, output_name_to_node)
            if is_same_root:
                mask_index = self.attention_mask.process_mask(squeeze_output_name)
                logger.debug("Create an Attention node.")
                attention_node = self.attention_fusion.create_attention_node(mask_index, matmul_q, matmul_k, matmul_v,
                                                                             add_q, add_k, add_v, parent.output[0],
                                                                             reshape_qkv.output[0])
                if parent.op_type == 'Reshape':
                    # Temporary work around: we require the skiplayernorm and attention op be fed with 3-d input
                    hidden_size = numpy_helper.to_array(self.get_initializer(parent.input[1]))[1]
                    tensor = helper.make_tensor(name=parent.name + "_modified",
                                                data_type=TensorProto.INT64,
                                                dims=[3],
                                                vals=np.int64([[1, -1, hidden_size]]).tobytes(),
                                                raw=True)
                    self.add_initializer(tensor)
                    parent.input[1] = parent.name + "_modified"

                if attention_node is None:
                    continue

                self.add_node(attention_node)
                attention_count += 1

                nodes_to_remove.extend([reshape_qkv, transpose_qkv, matmul_qkv])
                nodes_to_remove.extend(qk_nodes)
                nodes_to_remove.extend(q_nodes)
                nodes_to_remove.extend(k_nodes)
                nodes_to_remove.extend(v_nodes)
                nodes_to_remove.extend(mask_nodes)
            else:
                logger.debug("Root node not matched.")
                continue
        self.remove_nodes(nodes_to_remove)
        self.update_graph()
        logger.info(f"Fused Attention count:{attention_count}")

    def preprocess(self):
        self.remove_identity()
        self.process_embedding()
        #TODO: remove fuse mask since we have embedding fused so fuse_attention shall handle the mask nodes.
        self.fuse_mask()

    def remove_reshape_before_first_attention(self):
        attention_nodes = self.get_nodes_by_op_type("Attention")
        for attention_node in attention_nodes:
            path = self.match_parent_path(attention_node, ['Reshape', 'EmbedLayerNormalization'], [0, 0])
            if path is None:
                continue
            logger.info("Remove Reshape before first Attention node.")
            reshape, embed = path
            self.replace_input_of_all_nodes(reshape.output[0], reshape.input[0])
            self.remove_node(reshape)
            break

    def postprocess(self):
        self.remove_reshape_before_first_attention()
        # Temporary work around for the following comment as it will cause topological issues for a bert model
        # self.prune_graph()
