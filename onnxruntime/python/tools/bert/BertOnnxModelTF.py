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
    def __init(self, model, num_heads, hidden_size, sequence_length, input_int32, float16, gpu_only, verbose):
        super().__init__(model, num_heads, hidden_size, sequence_length, verbose)

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

    def __fuse_reshape_after_qkv(self, reshape_node, nodes_to_remove, nodes_to_add):
        path0 = self.match_parent_path(reshape_node, ['Cast', 'Concat'], [1, 0])
        if path0 is None:
            return
        cast_node, concat_node = path0

        if not len(concat_node.input) == 4:
            return

        shape = [0,0]

        if self.get_initializer(concat_node.input[2]) and self.get_initializer(concat_node.input[3]):
            concat_2 = self.get_initializer(concat_node.input[2])
            concat_3 = self.get_initializer(concat_node.input[3])
            shape.extend(numpy_helper.to_array(concat_2))
            shape.extend(numpy_helper.to_array(concat_3))
        else:
            return

        shape_value = np.asarray(shape, dtype=np.int64)

        constant_shape_name = self.create_node_name('Constant', 'constant_shape')
        new_node = onnx.helper.make_node('Constant',
            inputs=[],
            outputs=[constant_shape_name],
            value=onnx.helper.make_tensor(name='const_tensor',
                data_type=TensorProto.INT64,
                dims=shape_value.shape,
                vals=shape_value))
        reshape_node.input[1] = constant_shape_name
        reshape_node.name = self.create_node_name('Reshape', 'Reshape_Fuse')
        nodes_to_remove.extend([cast_node, concat_node])
        nodes_to_add.append(new_node)

    def __fuse_reshape_after_sotfmax(self, reshape_node, nodes_to_remove, nodes_to_add):
        # Check that it is reshape after softmax.
        path = self.match_parent_path(reshape_node, ['Transpose', 'MatMul', 'Softmax', 'Add'], [0, 0, 0, 0])
        if path is None:
            return

        path0 = self.match_parent_path(reshape_node, ['Cast', 'Concat', 'Unsqueeze', 'Mul'], [1, 0, 0, 0])
        if path0 is None:
            return
        cast_node, concat_node, unsqueeze_node, mul_node = path0

        # Verify that cast has attribute "to" = 7
        is_good_cast = False
        for att in cast_node.attribute:
            if att.name == 'to' and att.i == 7:
                is_good_cast = True
                break
        if not is_good_cast:
            return

        if not len(concat_node.input) == 2:
            return

        shape = [0,0]

        if self.get_initializer(concat_node.input[1]):
            concat_1 = self.get_initializer(concat_node.input[1])
            shape.extend(numpy_helper.to_array(concat_1))
        else:
            return

        shape_value = np.asarray(shape, dtype=np.int64)

        constant_shape_name = self.create_node_name('Constant', 'constant_shape')
        new_node = onnx.helper.make_node('Constant',
            inputs=[],
            outputs=[constant_shape_name],
            value=onnx.helper.make_tensor(name='const_tensor',
                data_type=TensorProto.INT64,
                dims=shape_value.shape,
                vals=shape_value))
        reshape_node.input[1] = constant_shape_name
        reshape_node.name = self.create_node_name('Reshape', 'Reshape_Fuse')
        nodes_to_remove.extend([cast_node, concat_node, unsqueeze_node, mul_node])
        nodes_to_add.append(new_node)

    def __fuse_reshape_after_normalize(self, reshape_node, nodes_to_remove):
        parent = self.get_parent(reshape_node, 0)
        if parent is None or not parent.op_type == self.normalize_name:
            return

        parent_path = self.match_parent_path(
            reshape_node,
            ['Cast', 'Concat', 'Unsqueeze', 'Cast', 'Squeeze', 'Slice', 'Cast', 'Shape'],
            [1,             0,           0,      0,         0,       0,       0,      0])

        if not parent_path is None:
            nodes_to_remove.extend(parent_path)

        nodes_to_remove.append(reshape_node)

        self.replace_input_of_all_nodes(reshape_node.output[0], reshape_node.input[0])




    def fuse_reshape(self):
        nodes = self.nodes()
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        nodes_to_add = []

        for reshape_node in self.get_nodes_by_op_type('Reshape'):
            self.__fuse_reshape_after_qkv(reshape_node, nodes_to_remove, nodes_to_add)
            self.__fuse_reshape_after_sotfmax(reshape_node, nodes_to_remove, nodes_to_add)
            self.__fuse_reshape_after_normalize(reshape_node, nodes_to_remove)

        logger.info(f"Count of nodes removed for Reshape fuse:{len(nodes_to_remove)}")

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
        skip_layernorm_nodes = []
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

                if root_node.op_type == 'Add':
                    subgraph_nodes.append(root_node)
                    if not self.is_safe_to_fuse_nodes(subgraph_nodes, node.output, self.input_name_to_nodes(), self.output_name_to_node()):
                        subgraph_nodes.pop()
                    else:
                        nodes_to_remove.append(root_node)
                        normalize_node = onnx.helper.make_node(self.normalize_name,
                            inputs=[root_node.input[0], root_node.input[1], weight_input, bias_input],
                            outputs=[node.output[0]],
                            name=self.create_node_name(self.normalize_name, name_prefix="SkipLayerNorm"))
                        normalize_node.domain = "com.microsoft"
                        skip_layernorm_nodes.extend([normalize_node])
                        continue

                normalize_node = onnx.helper.make_node('LayerNormalization',
                    inputs=[reduce_mean_node_1.input[0], weight_input, bias_input],
                    outputs=[node.output[0]], epsilon=epsilon)
                layernorm_nodes.extend([normalize_node])

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(skip_layernorm_nodes)
        self.add_nodes(layernorm_nodes)
        logger.info(f"Fused SkipLayerNormalization count: {len(skip_layernorm_nodes)}")
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

    def fuse_word_embedding(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == 'Reshape':
                data_path = self.match_parent_path(node, ['Gather', 'Reshape', 'Reshape'], [0, 1, 0])
                if data_path is None:
                    continue
                gather_node, reshape_node_0, reshape_node_1 = data_path
                shape_path = self.match_parent_path(
                    node,
                    ['Cast', 'Concat', 'Unsqueeze', 'Cast', 'Squeeze', 'Slice', 'Cast', 'Shape', 'Reshape'],
                    [     1,        0,           0,      0,         0,       0,      0,       0,         0])

                if shape_path is None:
                    continue

                cast_node_0, concat_node, unsqueeze_node, cast_node_1, squeeze_node, slice_node, cast_node_2, shape_node, reshape_node_2 = shape_path

                if not reshape_node_1 == reshape_node_2:
                    continue
                gather_node.input[1] = reshape_node_1.input[0]
                self.replace_input_of_all_nodes(node.output[0], gather_node.output[0])
                nodes_to_remove.extend([reshape_node_0, cast_node_0, concat_node, unsqueeze_node, cast_node_1, squeeze_node, slice_node, cast_node_2, shape_node, reshape_node_2, node])

        self.remove_nodes(nodes_to_remove)
        logger.info("Fused word embedding" if len(nodes_to_remove) > 0 else "Failed to fuse word embedding")

    def fuse_segment_embedding(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == 'Reshape':
                data_path = self.match_parent_path(node, ['MatMul', 'OneHot', 'Reshape'], [0, 0, 0])
                if data_path is None:
                    continue

                matmul_node, onehot_node, reshape_node_0 = data_path

                subgraph_nodes = [matmul_node, onehot_node, reshape_node_0]
                if self.get_initializer(onehot_node.input[2]) is None:
                    concat_node_0 = self.get_parent(onehot_node, 2)
                    if concat_node_0 is None or concat_node_0.op_type != 'Concat':
                        continue
                    subgraph_nodes.append(concat_node_0)
                    
                shape_path = self.match_parent_path(
                    node,
                    ['Cast', 'Concat', 'Unsqueeze', 'Cast', 'Squeeze', 'Slice', 'Cast', 'Shape', 'Gather'],
                    [     1,        0,           0,      0,         0,       0,      0,       0,         0])

                if shape_path is None:
                    continue

                cast_node_0, concat_node_1, unsqueeze_node, cast_node_1, squeeze_node, slice_node, cast_node_2, shape_node, gather_node = shape_path

                gather_node = onnx.helper.make_node(
                    'Gather',
                    inputs=[matmul_node.input[1], reshape_node_0.input[0]],
                    outputs=node.output,
                    name='segment_embedding_gather')

                nodes_to_remove.extend(subgraph_nodes)

                nodes_to_remove.extend([cast_node_0, concat_node_1, unsqueeze_node, cast_node_1, squeeze_node, slice_node, cast_node_2, shape_node, node])
                self.add_node(gather_node)

        self.remove_nodes(nodes_to_remove)
        logger.info("Fused segment embedding" if len(nodes_to_remove) > 0 else "Failed to fuse segment embedding")

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
        logger.info("Fused mask" if len(nodes_to_remove) > 0 else "Failed to fuse mask")

    def preprocess(self):
        self.remove_identity()
        self.fuse_word_embedding()
        self.fuse_segment_embedding()
        self.fuse_mask()
