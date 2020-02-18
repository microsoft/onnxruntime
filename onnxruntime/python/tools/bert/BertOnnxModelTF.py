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

class BertOnnxModelTF(BertOnnxModel):
    def __init(self, model, num_heads, hidden_size, sequence_length, input_int32, float16, gpu_only):
        super().__init__(model, num_heads, hidden_size, sequence_length)

    """
     Fuse Gelu with Erf into one node:
                   +----------------------------------------------+
                   |                                              |
                   |                                              v
                [root] --> Mul -----> Erf    -->   Add --> Mul -->Mul
                           (A=0.7071067690849304)  (B=1)  (B=0.5)

     Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
    """
    def fuse_gelu_with_elf(self, gelu_op_name):
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
            mul_half = children[0]
            
            if not self.has_constant_input(mul_half, 0.5):
                continue

            first_mul = self.match_parent(erf_node, 'Mul', 0, output_name_to_node)
            if first_mul is None:
                continue

            i = self.find_constant_input(first_mul, 0.7071067690849304, delta=0.001)
            if i < 0:
                continue

            root_node = self.get_parent(first_mul, 0 if i == 1 else 1, output_name_to_node)
            if root_node is None:
                continue

            if mul_half.output[0] not in input_name_to_nodes:
                continue
            children = input_name_to_nodes[mul_half.output[0]]
            if len(children) != 1 or children[0].op_type != 'Mul':
                continue
            last_mul = children[0]

            if not (last_mul.input[0] == root_node.output[0] or last_mul.input[1] == root_node.output[0]):
                continue

            subgraph_nodes = [first_mul, erf_node, add_after_erf, mul_half, last_mul]
            if not self.is_safe_to_fuse_nodes(subgraph_nodes, [last_mul.output[0]], input_name_to_nodes, output_name_to_node):
                continue

            nodes_to_remove.extend(subgraph_nodes)
            gelu_node = onnx.helper.make_node(gelu_op_name,
                inputs=[root_node.output[0]],
                outputs=[last_mul.output[0]])
            gelu_node.domain = "com.microsoft"
            nodes_to_add.append(gelu_node)

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)
        if len(nodes_to_add) > 0:
            logger.info("Fused {} count:{}".format('FastGelu (approximation)' if gelu_op_name == 'FastGelu' else 'Gelu', len(nodes_to_add)))


    """
     Fuse Gelu with tanh into one node:
          +---------------------------+
          |                           |
          |                           v
        [root] --> Pow --> Mul -----> Add  --> Mul --> Tanh --> Add --> Mul(B=0.5)-->Mul-->
          |       (Y=3)   (B=0.0447...)       (B=0.7978...)    (B=1)                  ^
          |                                                                           |
          +---------------------------------------------------------------------------+
     Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
    """
    def fuse_gelu_with_tanh(self, gelu_op_name):
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        nodes_to_add = []

        for node in self.get_nodes_by_op_type('Tanh'):
            tanh_node = node

            if node.output[0] not in input_name_to_nodes:
                continue
            children = input_name_to_nodes[node.output[0]]
            if len(children) != 1 or children[0].op_type != 'Add':
                continue
            add_after_tanh = children[0]

            if not self.has_constant_input(add_after_tanh, 1.0):
                continue

            if add_after_tanh.output[0] not in input_name_to_nodes:
                continue
            children = input_name_to_nodes[add_after_tanh.output[0]]
            if len(children) != 1 or children[0].op_type != 'Mul':
                continue
            mul_half = children[0]

            i = self.find_constant_input(mul_half, 0.5)
            if i < 0:
                continue

            if mul_half.output[0] not in input_name_to_nodes:
                continue
            children = input_name_to_nodes[mul_half.output[0]]
            if len(children) != 1 or children[0].op_type != 'Mul':
                continue
            mul_after_mul_half = children[0]

            root_node = self.get_parent(mul_after_mul_half, 0 if mul_after_mul_half.input[1] == mul_half.output[0] else 1, output_name_to_node)
            if root_node is None:
                continue

            mul_before_tanh = self.match_parent(tanh_node, 'Mul', 0, output_name_to_node)
            if mul_before_tanh is None:
                continue

            i = self.find_constant_input(mul_before_tanh, 0.7978, delta=0.0001)
            if i < 0:
                continue

            add_before_tanh = self.match_parent(mul_before_tanh, 'Add', 0 if i == 1 else 1, output_name_to_node)
            if add_before_tanh is None:
                continue

            mul_after_pow = self.match_parent(add_before_tanh, 'Mul', None, output_name_to_node, exclude=[root_node])
            if mul_after_pow is None:
                continue

            i = self.find_constant_input(mul_after_pow, 0.0447, delta=0.0001)
            if i < 0:
                continue

            pow = self.match_parent(mul_after_pow, 'Pow', 0 if i == 1 else 1, output_name_to_node)
            if pow is None:
                continue

            if not self.has_constant_input(pow, 3.0):
                continue

            if pow.input[0] != root_node.output[0]:
                continue

            subgraph_nodes = [mul_after_mul_half, mul_half, add_after_tanh, tanh_node, mul_before_tanh, add_before_tanh, mul_after_pow, pow]
            if not self.is_safe_to_fuse_nodes(subgraph_nodes, [mul_after_mul_half.output[0]], input_name_to_nodes, output_name_to_node):
                continue

            nodes_to_remove.extend(subgraph_nodes)
            gelu_node = onnx.helper.make_node(gelu_op_name,
                inputs=[root_node.output[0]],
                outputs=mul_after_mul_half.output,
                name=self.create_node_name(gelu_op_name))
            gelu_node.domain = "com.microsoft"
            nodes_to_add.append(gelu_node)

        if len(nodes_to_add) > 0:
            logger.info("Fused {} count: {}".format('Gelu (FastGelu fits better)' if gelu_op_name == 'Gelu' else 'FastGelu', len(nodes_to_add)))

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)

    """
      Batch Layer Norm from Keras in Tensorflow:
         +----------------------+
         |                      |
         |                      v                               (B)                             (B)             (A)
        Add --> ReduceMean -->  Sub  --> Mul --> ReduceMean --> Add --> Sqrt --> Reciprocol --> Mul --> Mul --> Sub --> Add
         |          |                                                                            |       ^              ^
         |          |                                                                            |       |              |
         |          +----------------------------------------------------------------------------|-------+              |
         |                                                                                       v                      |
         +-------------------------------------------------------------------------------------> Mul--------------------+
    """
    def fuse_layer_norm(self):
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        layernorm_nodes = []
        for node in self.nodes():
            if node.op_type == 'Add':
                return_indice=[]
                parent_nodes = self.match_parent_path(
                    node,
                    ['Sub', 'Mul', 'Mul', 'Reciprocal', 'Sqrt', 'Add', 'ReduceMean', 'Mul', 'Sub', 'ReduceMean'],
                    [   1,     1,   None,            0,      0,     0,         None,     0,    0,          None],
                    output_name_to_node,
                    return_indice=return_indice)

                if parent_nodes is None:
                    continue

                assert len(return_indice) == 3
                if not (return_indice[0] in [0, 1] and return_indice[1] in [0, 1] and return_indice[2] in [0, 1]):
                    logger.debug("return indice is exepected in [0, 1], but got {return_indice}")
                    continue

                sub_node_0, mul_node_0, mul_node_1, reciprocol_node, sqrt_node, add_node_0, reduce_mean_node_0, mul_node_2, sub_node_1, reduce_mean_node_1 = parent_nodes

                mul_node_3 = self.match_parent(node, 'Mul', 0, output_name_to_node)
                if mul_node_3 is None:
                    logger.debug("mul_node_3 not found")
                    continue

                root_node = self.get_parent(reduce_mean_node_1, 0, output_name_to_node)
                if root_node is None:
                    logger.debug("root node is none")
                    continue

                i, epsilon = self.get_constant_input(add_node_0)
                if epsilon is None or epsilon <= 0 or epsilon > 1.0E-5:
                    logger.debug("epsilon is not matched")
                    continue

                if reduce_mean_node_1.input[0] not in mul_node_3.input or reduce_mean_node_1.input[0] not in sub_node_1.input:
                    logger.debug("reduce_mean_node_1 and mul_node_3 shall link from root node")
                    continue

                if mul_node_2.input[0] != mul_node_2.input[1]:
                    logger.debug("mul_node_2 shall have two same inputs")
                    continue

                subgraph_nodes = [node, sub_node_0, mul_node_0, mul_node_1, reciprocol_node, sqrt_node, add_node_0, reduce_mean_node_0, mul_node_2, sub_node_1, reduce_mean_node_1,mul_node_3]
                if not self.is_safe_to_fuse_nodes(subgraph_nodes, node.output, self.input_name_to_nodes(), self.output_name_to_node()):
                    logger.debug("not safe to fuse layer normalization")
                    continue

                nodes_to_remove.extend(subgraph_nodes)

                weight_input = mul_node_1.input[1]
                bias_input = sub_node_0.input[0]

                normalize_node = onnx.helper.make_node(
                    'LayerNormalization',
                    inputs=[reduce_mean_node_1.input[0], weight_input, bias_input],
                    outputs=[node.output[0]])
                normalize_node.attribute.extend([onnx.helper.make_attribute("epsilon", float(epsilon))])
                layernorm_nodes.extend([normalize_node])

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(layernorm_nodes)
        logger.info(f"Fused LayerNormalization count: {len(layernorm_nodes)}")

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
                    [        1,     0,                 0,      0,        0,           0,      0,         0,        0,     0,       0])
                if parent_path_constant is None:
                    continue
                reshape_node_0, mul_node_0, constantofshape_node, cast_node_0, concat_node_0, unsqueeze_node, cast_node_1, squeeze_node, slice_node, cast_node_2, shape_node = parent_path_constant

                parent_path_mask = self.match_parent_path(
                    mul_node_0,
                    ['Cast', 'Reshape', 'Cast', 'Concat', 'Unsqueeze'],
                    [     1,     0,          1,       0,            0])

                if parent_path_mask is None:
                    continue

                cast_node_3, reshape_node_1, cast_node_4, concat_node_1, unsqueeze_node_1 = parent_path_mask

                if not unsqueeze_node_1 == unsqueeze_node:
                    continue

                unsqueeze_added_1 = onnx.helper.make_node(
                    'Unsqueeze',
                    inputs=[reshape_node_1.input[0]],
                    outputs=['mask_fuse_unsqueeze1_output'],
                    name='Mask_UnSqueeze_1',
                    axes=[1])

                unsqueeze_added_2 = onnx.helper.make_node(
                    'Unsqueeze',
                    inputs=['mask_fuse_unsqueeze1_output'],
                    outputs=[cast_node_3.input[0]],
                    name='Mask_UnSqueeze_2',
                    axes=[2])
                node.input[1] = cast_node_3.output[0]

                nodes_to_remove.extend([reshape_node_0, mul_node_0, constantofshape_node, cast_node_0, concat_node_0, unsqueeze_node, cast_node_1, squeeze_node, slice_node, cast_node_2, shape_node])
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
                mask_path = self.match_parent_path(
                    node,
                    ['Sub', 'Unsqueeze', 'Mul', 'Cast', 'Reshape', 'Cast'],
                    [  0,            1,     0,      1,         0,      0])
                if mask_path is None:
                    continue
                sub_node, unsqueeze_node, mul_node, cast_node_0, reshape_node_0, cast_node_1 = mask_path

                mask_input_name = next(iter(self.mask_indice))
                if cast_node_1.input[0] != mask_input_name:
                    print("Cast input {} is not mask input{}".format(cast_node_1.input[0], mask_input_name))
                    continue

                unsqueeze_added_1 = onnx.helper.make_node(
                    'Unsqueeze',
                    inputs=[mask_input_name],
                    outputs=['mask_fuse_unsqueeze1_output'],
                    name='Mask_UnSqueeze_1',
                    axes=[1])

                unsqueeze_added_2 = onnx.helper.make_node(
                    'Unsqueeze',
                    inputs=['mask_fuse_unsqueeze1_output'],
                    outputs=['mask_fuse_unsqueeze2_output'],
                    name='Mask_UnSqueeze_2',
                    axes=[2])

                cast_node_2 = onnx.helper.make_node(
                    'Cast',
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

    def find_segment_ids(self, segment_embedding):
        input_name_to_nodes = self.input_name_to_nodes()
        if segment_embedding not in input_name_to_nodes:
            return None

        nodes = input_name_to_nodes[segment_embedding]
        if len(nodes) != 1:
            return None

        graph_inputs = self.get_graph_inputs(nodes[0], recursive=True)
        if len(graph_inputs) == 1:
            return graph_inputs[0]

        print("Found multiple candidates of segment_ids", graph_inputs)
        return None

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
                mask_path = self.match_parent_path(
                        node,
                        ['Add', 'Mul', 'Sub'],
                        [ 0,     1,     None])
                if mask_path is None:
                    continue
                add_node, mul_node, sub_node = mask_path
                if self.has_constant_input(mul_node, -10000) and self.has_constant_input(sub_node, 1):
                    graph_inputs = self.get_graph_inputs(sub_node, recursive=True)
                    inputs = [input for input in graph_inputs if input not in excluded_graph_inputs]
                    if len(inputs) == 1:
                        return inputs[0]

        return None

    def create_embedding_subgraph(self, normalize_node, word_embedding, segment_embedding, position_embedding):
        segment_ids = self.find_segment_ids(segment_embedding)
        if segment_ids is None:
            logger.info("Failed to find segment_ids. Cannot fuse embedding layer.")
            return False

        input_ids = self.find_input_ids(word_embedding)
        if input_ids is None:
            logger.info("Failed to find input_ids. Cannot fuse embedding layer.")
            return False

        mask_input = self.find_mask_input([segment_ids, input_ids])
        if mask_input is None:
            logger.info("Failed to find input_mask. Cannot fuse embedding layer.")
            return False

        self.bert_inputs = [input_ids, segment_ids, mask_input]

        mask_index = self.create_node_name('mask_index')
        self.mask_indice[mask_input] = mask_index

        if self.find_graph_input(input_ids).type.tensor_type.elem_type != TensorProto.INT32:
            casted, input_ids = self.cast_graph_input_to_int32(input_ids)

        if self.find_graph_input(segment_ids).type.tensor_type.elem_type != TensorProto.INT32:
            casted, segment_ids = self.cast_graph_input_to_int32(segment_ids)

        if self.find_graph_input(mask_input).type.tensor_type.elem_type != TensorProto.INT32:
            casted, mask_input = self.cast_graph_input_to_int32(mask_input)

        embed_output = self.create_node_name('embed_output')
        embed_node = onnx.helper.make_node('EmbedLayerNormalization',
                        inputs=[input_ids,
                                segment_ids, 
                                word_embedding,
                                position_embedding,
                                segment_embedding,
                                normalize_node.input[1], # gamma
                                normalize_node.input[2], # beta
                                mask_input],
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
            pos_embed_path = self.match_parent_path(
                layer_norm_node,
                ['Add', 'Reshape', 'Slice'],
                [ 0,     1,         0],
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
                    logger.warning("Failed to find two embeddings (word and segment) from Add node. Found {}".format(embeddings))
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

    def preprocess(self):
        self.remove_identity()
        self.process_embedding()
        #TODO: remove fuse mask since we have embedding fused so fuse_attention shall handle the mask nodes.
        self.fuse_mask()

    def remove_reshape_before_first_attention(self):
        attention_nodes = self.get_nodes_by_op_type("Attention")
        for attention_node in attention_nodes:
            path = self.match_parent_path(
                        attention_node,
                        ['Reshape', 'EmbedLayerNormalization'],
                        [ 0,     0])
            if path is None:
                continue
            logger.info("Remove Reshape before first Attention node.")
            reshape, embed = path
            self.replace_input_of_all_nodes(reshape.output[0], reshape.input[0])
            self.remove_node(reshape)
            break

    def postprocess(self):
        self.remove_reshape_before_first_attention()
        self.prune_graph()
