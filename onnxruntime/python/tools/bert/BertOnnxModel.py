#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import onnx
import sys
import argparse
import numpy as np
from collections import deque
from onnx import ModelProto, TensorProto, numpy_helper
from OnnxModel import OnnxModel

class BertOnnxModel(OnnxModel):
    def __init__(self, model, num_heads, hidden_size, sequence_length, input_int32, float16, gpu_only, verbose):
        assert num_heads > 0
        assert hidden_size % num_heads == 0
        assert sequence_length > 0
        
        super(BertOnnxModel, self).__init__(model, verbose)
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

        self.input_int32 = input_int32
        self.gpu_only = gpu_only
        self.float16 = float16

        # A lookup table with mask input as key, and mask index output as value
        self.mask_indice = {}
        # A lookup table with mask input as key, and cast (to int32) output as value
        self.mask_casted = {}

        self.bert_inputs = []

        # constant node names
        self.normalize_name = "SkipLayerNormalization"
        self.attention_name = 'Attention'

    def get_normalize_nodes(self):
        return self.get_nodes_by_op_type(self.normalize_name)

    def normalize_children_types(self):
        return ['MatMul', 'MatMul', 'MatMul', 'SkipLayerNormalization']

    def cast_graph_input_to_int32(self, input_name):
        graph_input = self.find_graph_input(input_name)
        if graph_input is not None and graph_input.type.tensor_type.elem_type != TensorProto.INT32:
            cast_output = input_name + '_int32'
            cast_node = onnx.helper.make_node('Cast', inputs=[input_name], outputs=[cast_output])
            cast_node.attribute.extend([onnx.helper.make_attribute("to", int(TensorProto.INT32))])
            self.add_node(cast_node)
            return True, cast_output

        return False, input_name

    def undo_cast_input_to_int32(self, input_name):
        input_name_to_nodes = self.input_name_to_nodes()
        nodes =  input_name_to_nodes[input_name]
        for node in nodes:
            if node.op_type == "Cast":
                is_int32 = False
                for att in node.attribute:
                    if att.name == 'to' and att.i == int(TensorProto.INT32):
                        is_int32 = True
                        break
                if is_int32:
                    output_name = node.output[0]
                    self.remove_node(node)
                    self.replace_input_of_all_nodes(output_name, input_name)

    def process_mask(self, input):
        if input in self.mask_indice:
            return self.mask_indice[input]

        # Add cast to convert int64 to int32
        casted, input_name = self.cast_graph_input_to_int32(input)
        if casted:
            self.mask_casted[input] = input_name

        # Add a mask processing node
        output_name = self.create_node_name('mask_index')
        mask_index_node = onnx.helper.make_node('ReduceSum',
            inputs=[input_name],
            outputs=[output_name],
            name=self.create_node_name('ReduceSum', 'MaskReduceSum'))
        mask_index_node.attribute.extend([onnx.helper.make_attribute("axes", [1]), onnx.helper.make_attribute("keepdims", 0)])
        self.add_node(mask_index_node)
        
        self.mask_indice[input] = output_name
        return output_name

    def create_attention_node(self, mask_index, q_matmul, k_matmul, v_matmul, q_add, k_add, v_add, input, output):
        q_weight = self.get_initializer(q_matmul.input[1])
        k_weight = self.get_initializer(k_matmul.input[1])
        v_weight = self.get_initializer(v_matmul.input[1])
        q_bias = self.get_initializer(q_add.input[1])
        k_bias = self.get_initializer(k_add.input[1])
        v_bias = self.get_initializer(v_add.input[1])
        
        qw = numpy_helper.to_array(q_weight)
        assert qw.shape == (self.hidden_size, self.hidden_size)

        kw = numpy_helper.to_array(k_weight)
        assert kw.shape == (self.hidden_size, self.hidden_size)

        vw = numpy_helper.to_array(v_weight)
        assert vw.shape == (self.hidden_size, self.hidden_size)

        qkv_weight = np.stack((qw, kw, vw), axis=-2)
        
        qb = numpy_helper.to_array(q_bias)
        assert qb.shape == (self.hidden_size,)

        kb = numpy_helper.to_array(k_bias)
        assert kb.shape == (self.hidden_size,)

        vb = numpy_helper.to_array(v_bias)
        assert vb.shape == (self.hidden_size,)

        qkv_bias = np.stack((qb, kb, vb), axis=-2)

        attention_node_name = self.create_node_name(self.attention_name)

        weight = onnx.helper.make_tensor(name=attention_node_name + '_qkv_weight',
            data_type=TensorProto.FLOAT,
            dims=[self.hidden_size, 3 * self.hidden_size],
            vals=qkv_weight.flatten().tolist())
        self.add_initializer(weight)

        weight_input = onnx.helper.make_tensor_value_info(weight.name, TensorProto.FLOAT, [self.hidden_size, 3 * self.hidden_size])
        self.add_input(weight_input)

        bias = onnx.helper.make_tensor(name=attention_node_name + '_qkv_bias',
            data_type=TensorProto.FLOAT,
            dims=[3 * self.hidden_size],
            vals=qkv_bias.flatten().tolist())
        self.add_initializer(bias)

        bias_input = onnx.helper.make_tensor_value_info(bias.name, TensorProto.FLOAT, [3 * self.hidden_size])
        self.add_input(bias_input)

        attention_node = onnx.helper.make_node(self.attention_name,
            inputs=[input, attention_node_name + '_qkv_weight', attention_node_name + '_qkv_bias', mask_index],            outputs=[output],
            name=attention_node_name)
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([onnx.helper.make_attribute("num_heads", self.num_heads)])

        self.add_node(attention_node)

    def fuse_attention(self):
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        attention_count = 0

        for normalize_node in self.get_normalize_nodes():
            # SkipLayerNormalization has two inputs, and one of them is the
            # root input for attention.
            qkv_nodes = None
            root_input = None
            for i, input in enumerate(normalize_node.input):
                if input not in output_name_to_node:
                    continue
                children = input_name_to_nodes[input]
                children_types = sorted([child.op_type for child in children])
                if children_types != self.normalize_children_types():
                    qkv_nodes = self.match_parent_path(normalize_node, ['Add', 'MatMul', 'Reshape', 'Transpose', 'MatMul'], [i, 0, 0, 0, 0])
                else:
                    root_input = input

            if root_input is None or qkv_nodes is None:
                continue

            (add_qkv, matmul_qkv, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

            v_nodes = self.match_parent_path(matmul_qkv, ['Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0])
            if v_nodes is None:
                continue
            (transpose_v, reshape_v, add_v, matmul_v) = v_nodes

            qk_nodes = self.match_parent_path(matmul_qkv, ['Softmax', 'Add', 'Div', 'MatMul'], [0, 0, 0, 0])
            if qk_nodes is None:
                qk_nodes = self.match_parent_path(matmul_qkv, ['Softmax', 'Add', 'Mul', 'MatMul'], [0, 0, 0, 0])
                if qk_nodes is None:
                     continue
            (softmax_qk, add_qk, div_qk, matmul_qk) = qk_nodes

            q_nodes = self.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'MatMul'], [0, 0, 0, 0])
            if q_nodes is None:
                continue
            (transpose_q, reshape_q, add_q, matmul_q) = q_nodes

            k_nodes = self.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0])
            if k_nodes is None:
                k_nodes = self.match_parent_path(matmul_qk, ['Transpose', 'Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0, 0])
                if k_nodes is None:
                    continue
                (transpose_k, transpose_k_2, reshape_k, add_k, matmul_k) = k_nodes
            else:
                (transpose_k, reshape_k, add_k, matmul_k) = k_nodes

            mask_nodes = self.match_parent_path(add_qk, ['Mul', 'Sub', 'Cast', 'Unsqueeze', 'Unsqueeze'], [1, 0, 1, 0, 0])
            if mask_nodes is None:
                continue
            (mul_mask, sub_mask, cast_mask, unsqueeze_mask, unsqueeze_mask_0) = mask_nodes

            if matmul_v.input[0] == root_input and matmul_q.input[0] == root_input and matmul_v.input[0] == root_input:
                mask_index = self.process_mask(unsqueeze_mask_0.input[0])
                self.create_attention_node(mask_index, matmul_q, matmul_k, matmul_v, add_q, add_k, add_v, root_input, reshape_qkv.output[0])
                nodes_to_remove.extend([reshape_qkv, transpose_qkv, matmul_qkv])
                nodes_to_remove.extend(qk_nodes)
                nodes_to_remove.extend(q_nodes)
                nodes_to_remove.extend(k_nodes)
                nodes_to_remove.extend(v_nodes)
                nodes_to_remove.extend(mask_nodes)
                attention_count += 1

        self.remove_nodes(nodes_to_remove)
        self.update_graph()
        print("Fused Attention count:", attention_count)

    def fuse_gelu(self, gelu_op_name):
        self.fuse_gelu_with_elf(gelu_op_name)
        self.fuse_gelu_with_tanh(gelu_op_name)

    """
     Fuse Gelu with Erf into one node:
                   +-------Mul(B=0.5)-------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul -->
                          (B=1.4142...)       (B=1)

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
            mul_after_erf = children[0]

            div = self.match_parent(erf_node, 'Div', 0, output_name_to_node)
            if div is None:
                continue

            if self.find_constant_input(div, 1.4142, delta=0.001) != 1:
                continue

            root_node = self.get_parent(div, 0, output_name_to_node)
            if root_node is None:
                continue

            mul_half = self.match_parent(mul_after_erf, 'Mul', None, output_name_to_node)
            if mul_half is None:
                continue
            
            if not self.has_constant_input(mul_half, 0.5):
                continue

            subgraph_nodes = [div, erf_node, add_after_erf, mul_after_erf, mul_half]
            if not self.is_safe_to_fuse_nodes(subgraph_nodes, [mul_after_erf.output[0]], input_name_to_nodes, output_name_to_node):
                continue

            nodes_to_remove.extend(subgraph_nodes)
            gelu_node = onnx.helper.make_node(gelu_op_name,
                inputs=[root_node.output[0]],
                outputs=[mul_after_erf.output[0]])
            gelu_node.domain = "com.microsoft"
            nodes_to_add.append(gelu_node)

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)
        if len(nodes_to_add) > 0:
            print("Fused {} count:{}".format('FastGelu (approximation)' if gelu_op_name == 'FastGelu' else 'Gelu', len(nodes_to_add)))

    """
     Fuse Gelu with tanh into one node:
          +---------------------------+
          |                           |
          |                           v
        [root] --> Pow --> Mul -----> Add  --> Mul --> Tanh --> Add --> Mul
          |       (Y=3)   (B=0.0447...)       (B=0.7978...)    (B=1)     ^
          |                                                              |
          +------> Mul(B=0.5)--------------------------------------------+
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
            mul_after_tanh = children[0]

            mul_half = self.match_parent(mul_after_tanh, 'Mul', None, output_name_to_node)
            if mul_half is None:
                continue

            i = self.find_constant_input(mul_half, 0.5)
            if i < 0:
                continue

            root_node = self.get_parent(mul_half, 0 if i == 1 else 1, output_name_to_node)
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

            subgraph_nodes = [mul_after_tanh, mul_half, add_after_tanh, tanh_node, mul_before_tanh, add_before_tanh, mul_after_pow, pow]
            if not self.is_safe_to_fuse_nodes(subgraph_nodes, [mul_after_tanh.output[0]], input_name_to_nodes, output_name_to_node):
                continue

            nodes_to_remove.extend(subgraph_nodes)
            gelu_node = onnx.helper.make_node(gelu_op_name,
                inputs=[root_node.output[0]],
                outputs=mul_after_tanh.output,
                name=self.create_node_name(gelu_op_name))
            gelu_node.domain = "com.microsoft"
            nodes_to_add.append(gelu_node)

        if len(nodes_to_add) > 0:
            print("Fused {} count: {}".format('Gelu (FastGelu fits better)' if gelu_op_name == 'Gelu' else 'FastGelu', len(nodes_to_add)))

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)

    def fuse_add_bias_gelu(self):
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()
        nodes_to_remove = []
        nodes_to_add = []

        # Don't need to fuse Gelu+Add here because ORT native code can handle it
        for node in self.get_nodes_by_op_type('FastGelu'):
            if len(node.input) != 1:
                continue

            nodes = self.match_parent_path(node, ['Add', 'MatMul'], [0, None])
            if nodes is None:
                continue
            (add, matmul) = nodes

            # bias should be one dimension
            bias_index = -1
            for i, input in enumerate(add.input):
                initializer = self.get_initializer(input)
                if initializer is None:
                    continue
                bias_index = i
                bias_weight = numpy_helper.to_array(initializer)
                break
            if bias_weight is None:
                continue
            if len(bias_weight.shape) != 1:
                continue

            subgraph_nodes = [node, add]
            if not self.is_safe_to_fuse_nodes(subgraph_nodes, [node.output[0]], input_name_to_nodes, output_name_to_node):
                continue

            nodes_to_remove.extend(subgraph_nodes)
            gelu_node = onnx.helper.make_node('FastGelu',
                inputs=[matmul.output[0], add.input[bias_index]],
                outputs=node.output,
                name=self.create_node_name('FastGelu', "FastGelu_AddBias_"))
            gelu_node.domain = "com.microsoft"
            nodes_to_add.append(gelu_node)

        if len(nodes_to_add) > 0:
            print("Fused FastGelu with Bias count:", len(nodes_to_add))

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)

    def fuse_add_bias_skip_layer_norm(self):
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()
        nodes_to_remove = []
        nodes_to_add = []

        for node in self.get_normalize_nodes():
            if len(node.input) != 4:
                continue

            nodes = self.match_parent_path(node, ['Add', 'MatMul'], [0, None])
            if nodes is None:
                continue
            (add, matmul) = nodes

            # bias should be one dimension
            bias_index = -1
            for i, input in enumerate(add.input):
                initializer = self.get_initializer(input)
                if initializer is None:
                    continue
                bias_index = i
                bias_weight = numpy_helper.to_array(initializer)
                break
            if bias_weight is None:
                continue
            if len(bias_weight.shape) != 1:
                continue

            subgraph_nodes = [node, add]
            if not self.is_safe_to_fuse_nodes(subgraph_nodes, [node.output[0]], input_name_to_nodes, output_name_to_node):
                continue

            nodes_to_remove.extend(subgraph_nodes)
            new_node = onnx.helper.make_node(self.normalize_name,
                inputs=[matmul.output[0], node.input[1], node.input[2], node.input[3], add.input[bias_index]],
                outputs=node.output,
                name=self.create_node_name(self.normalize_name, self.normalize_name + "_AddBias_"))
            new_node.domain = "com.microsoft"
            nodes_to_add.append(new_node)

        if len(nodes_to_add) > 0:
            print("Fused SkipLayerNormalization with Bias count:", len(nodes_to_add))

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)

    def fuse_reshape(self):
        nodes = self.nodes()
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        nodes_to_add = []

        for reshape_node in self.get_nodes_by_op_type('Reshape'):
            if reshape_node.input[1] not in output_name_to_node:
                continue
            concat_node = output_name_to_node[reshape_node.input[1]]
            if concat_node.op_type != 'Concat' or len(concat_node.input) < 3 or len(concat_node.input) > 4:
                continue

            path0 = self.match_parent_path(concat_node, ['Unsqueeze', 'Gather', 'Shape'], [0, 0, 0], output_name_to_node)
            if path0 is None:
                continue
            (unsqueeze_0, gather_0, shape_0) = path0

            path1 = self.match_parent_path(concat_node, ['Unsqueeze', 'Gather', 'Shape'], [1, 0, 0], output_name_to_node)
            if path1 is None:
                continue
            (unsqueeze_1, gather_1, shape_1) = path1

            shape = []
            gather_value = self.get_constant_value(gather_0.input[1])
            if gather_value == 0:
                shape.append(0)

            gather_value = self.get_constant_value(gather_1.input[1])
            if gather_value == 1:
                shape.append(0)

            if len(shape) != 2:
                continue

            path2 = []
            path3 = []
            shape_nodes = [shape_0, shape_1]
            if len(concat_node.input) == 3 and self.get_initializer(concat_node.input[2]) is None:
                path2 = self.match_parent_path(concat_node, ['Unsqueeze', 'Mul', 'Gather', 'Shape'], [2, 0, 0, 0], output_name_to_node)
                path3 = self.match_parent_path(concat_node, ['Unsqueeze', 'Mul', 'Gather', 'Shape'], [2, 0, 1, 0], output_name_to_node)
                if path2 is None or path3 is None:
                    continue
                shape_nodes.extend([path2[-1], path3[-1]])
                shape.append(-1)
            elif (len(concat_node.input) > 2):
                concat_2 = self.get_initializer(concat_node.input[2])
                if concat_2 is None:
                    continue
                shape.extend(numpy_helper.to_array(concat_2))

            if len(concat_node.input) == 4 and self.get_initializer(concat_node.input[3]) is None:
                path2 = self.match_parent_path(concat_node, ['Unsqueeze', 'Div', 'Gather', 'Shape'], [3, 0, 0, 0], output_name_to_node)
                shape_nodes.extend([path2[-1]])
                if path2 is None or -1 in shape:
                    continue
                shape.append(-1)
            elif (len(concat_node.input) > 3):
                concat_3 = self.get_initializer(concat_node.input[3])
                if concat_3 is None:
                    continue
                shape.extend(numpy_helper.to_array(concat_3))

            root_input = reshape_node.input[0]
            same_shape_input = True
            for shape_node in shape_nodes:
                if shape_node.input[0] != root_input:
                    same_shape_input = False

            if not same_shape_input:
                continue

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
            nodes_to_remove.extend([concat_node])
            nodes_to_remove.extend(path0)
            nodes_to_remove.extend(path1)
            nodes_to_remove.extend(path2)
            nodes_to_remove.extend(path3)
            nodes_to_add.append(new_node)

        print("Fused Reshape count:", len(nodes_to_add))

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)

    """
     Embed Layer Normalization will fuse embeddings and mask processing into one node.
     The embeddings before conversion:

     (input_ids) -------->  Gather ----------+       (segment_ids)
        |                                    |            |
        |                                    v            v
        +--> Shape --> Expand -> Gather---->Add         Gather
        |                ^                   |            |
        |                |                   v            v
        +---(optional graph)               SkipLayerNormalization

      Optional graph is used to generate position list (0, 1, ...) per batch. It can be a constant in some model.

      (input_ids) --> Gather -----+           Slice
                                  |            |
                                  v            v
     (segment_ids)--> Gather --->Add        Reshape
                                  |            |
                                  v            v
                              SkipLayerNormalization
    """
    def fuse_embed_layer(self):
        nodes = self.nodes()
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        if len(self.mask_indice) == 0:
            print("skip embed layer fusion since mask input is not found")
            return
        if len(self.mask_indice) > 1:
            print("skip embed layer fusion since there are multiple mask inputs found")
            return
        mask_input_name = next(iter(self.mask_indice))
        mask_output_name = self.mask_indice[mask_input_name]
        mask_node = output_name_to_node[mask_output_name]

        nodes_to_remove = []

        # Find the first normalize node could be embedding layer.
        normalize_node = None
        for node in self.get_normalize_nodes():
            if self.match_parent_path(node, ['Add', 'Gather'], [0, 0]) is not None:
                if self.find_first_child_by_type(node, 'Attention', input_name_to_nodes, recursive=False) is not None:
                    normalize_node = node
                    break

        if normalize_node is None:
            print("Failed to find embedding layer")

        # Here we assume the order of embedding is word_embedding +
        # position_embedding + segment_embedding.
        word_embedding_path = self.match_parent_path(normalize_node, ['Add', 'Gather'], [0, 0])
        if word_embedding_path is None:
            print("Failed to find word embedding")
            return
        add_node, word_embedding_gather = word_embedding_path
        input_ids = word_embedding_gather.input[1]

        position_embedding_path = self.match_parent_path(normalize_node, ['Reshape', 'Slice'], [1, 0])

        position_embedding_expand = None
        if position_embedding_path is None:
            position_embedding_path = self.match_parent_path(add_node, ['Gather', 'Expand', 'Shape'], [1, 1, 1])
            if position_embedding_path is None:
                position_embedding_path = self.match_parent_path(add_node, ['Gather', 'Expand', 'Concat', 'Unsqueeze', 'Gather', 'Shape'], [1, 1, 1, 1, 0, 0])
                if position_embedding_path is None:
                    print("Failed to find position embedding")
                    return
                position_embedding_weight_node, position_embedding_expand, _, _, _, position_embedding_shape = position_embedding_path
            else:
                position_embedding_weight_node, position_embedding_expand, position_embedding_shape = position_embedding_path

            if not position_embedding_shape is None and position_embedding_shape.input[0] != input_ids:
                print("position and word embedding is expected to be applied on same input")
                return
        else:
            _, position_embedding_weight_node = position_embedding_path

        segment_embedding_path = self.match_parent_path(normalize_node, ['Gather'], [1])
        if segment_embedding_path is None:
            segment_embedding_path = self.match_parent_path(normalize_node, ['Add', 'Gather'], [0, 1])
            if segment_embedding_path is None:
                print("Failed to find segment embedding")
                return
            _, segment_embedding_gather = segment_embedding_path
        else:
            segment_embedding_gather = segment_embedding_path[0]

        segment_ids = segment_embedding_gather.input[1]


        if position_embedding_expand:
            subgraph_nodes = self.get_parent_subgraph_nodes(position_embedding_expand, [], output_name_to_node)
            nodes_to_remove.extend(subgraph_nodes)
        
        nodes_to_remove.extend(word_embedding_path)
        nodes_to_remove.extend(position_embedding_path)
        nodes_to_remove.extend(segment_embedding_path)

        nodes_to_remove.extend([normalize_node])
        nodes_to_remove.extend([mask_node])

        # store inputs for further processing
        self.bert_inputs = [input_ids, segment_ids, mask_input_name]

        if not self.input_int32:
            # When mask has been casted to int32, use that casted one as input of embed layer norm.
            if mask_input_name in self.mask_casted:
                mask_input_name = self.mask_casted[mask_input_name]

            # Cast input_ids and segment_ids to int32.
            casted, input_ids = self.cast_graph_input_to_int32(input_ids)

            casted, segment_ids = self.cast_graph_input_to_int32(segment_ids)
        else:
            self.undo_cast_input_to_int32(mask_input_name)

        embed_node = onnx.helper.make_node('EmbedLayerNormalization',
                        inputs=[input_ids,
                                segment_ids, 
                                word_embedding_gather.input[0],
                                position_embedding_weight_node.input[0],
                                segment_embedding_gather.input[0],
                                normalize_node.input[2],
                                normalize_node.input[3], # gamma and beta
                                mask_input_name],
                        outputs=["embed_output", mask_output_name],
                        name="EmbedLayer")

        embed_node.domain = "com.microsoft"
        
        self.replace_input_of_all_nodes(normalize_node.output[0], 'embed_output')

        self.remove_nodes(nodes_to_remove)
        self.add_node(embed_node)
        self.update_graph()
        print("Fused EmbedLayerNormalization count: 1")

        # Change graph input data type int32 if needed.
        if self.input_int32:
            self.change_input_to_int32()

    def get_bert_inputs(self, include_mask=True):
        return self.bert_inputs if include_mask else self.bert_inputs[:2]

    def get_batch_size_from_graph_input(self):
        graph = self.graph()
        bert_inputs = self.get_bert_inputs()
        for input in graph.input:
            if input.name in bert_inputs:
                tensor_type = input.type.tensor_type
                if (tensor_type.HasField("shape")):
                    for d in tensor_type.shape.dim:
                        if (d.HasField("dim_value")):
                            return d.dim_value
                        elif (d.HasField("dim_param")):
                            return str(d.dim_param)       # unknown dimension with symbolic name
                        return None
        return None

    def change_input_to_int32(self):
        original_opset_version = self.model.opset_import[0].version
        graph = self.graph()

        batch_size = self.get_batch_size_from_graph_input()
        input_batch_size = batch_size if isinstance(batch_size, int) else 1
        new_graph_inputs = []

        bert_inputs = self.get_bert_inputs()
        for input in graph.input:
            if input.name in bert_inputs:
                int32_input = onnx.helper.make_tensor_value_info(input.name, TensorProto.INT32, [input_batch_size, self.sequence_length])
                new_graph_inputs.append(int32_input)
            else:
                new_graph_inputs.append(input)

        graph_def = onnx.helper.make_graph(graph.node,
                                           'int32 inputs',
                                           new_graph_inputs,
                                           graph.output,
                                           initializer=graph.initializer,
                                           value_info=graph.value_info)

        self.model = onnx.helper.make_model(graph_def, producer_name='bert model optimizer')

        if isinstance(batch_size, str):
            self.update_dynamic_batch_io(batch_size)

        # restore opset version
        self.model.opset_import[0].version = original_opset_version


    # Update input and output using dynamic batch
    def update_dynamic_batch_io(self, dynamic_batch_dim='batch'):

        bert_inputs = self.get_bert_inputs()
        dynamic_batch_inputs = {}
        for input in self.model.graph.input:
            for bert_input in bert_inputs:
                if bert_input == input.name:
                    dim_proto = input.type.tensor_type.shape.dim[0]
                    dim_proto.dim_param = dynamic_batch_dim

        for output in self.model.graph.output:
            dim_proto = output.type.tensor_type.shape.dim[0]
            dim_proto.dim_param = dynamic_batch_dim

    """
     Layer Normalization will fuse Add + LayerNormalization into one node:
          +----------------------+
          |                      |
          |                      v
        Add --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                 (axis=2 or -1)  |      (Y=2)   (axis=2 or -1)  (E-6 or E-12 or 0)    ^
                                 |                                               |
                                 +-----------------------------------------------+

     It also handles cases of duplicated sub nodes exported from older version of PyTorch:
          +----------------------+
          |                      v
          |           +-------> Sub-----------------------------------------------+
          |           |                                                           |
          |           |                                                           v
        Add --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div  --> Mul --> Add
          |                      ^
          |                      |
          +----------------------+

      TODO: Batch Layer Norm from Keras in Tensorflow:
         +----------------------+
         |                      |
         |                      v                                                                      (B)     (A)
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
            if node.op_type == 'ReduceMean':
                children = self.get_children(node, input_name_to_nodes)
                if len(children) == 0 or len(children) > 2:
                    continue

                parent = self.get_parent(node, 0, output_name_to_node)
                if parent is None:
                    continue

                if children[0].op_type != 'Sub' or self.get_parent(children[0], 0, output_name_to_node) != parent:
                    continue

                if len(children) == 2:
                    if children[0].op_type != 'Sub' or self.get_parent(children[1], 0, output_name_to_node) != parent:
                        continue

                div_node = None
                for child in children:
                    if child.op_type == 'Sub':
                        div_node = self.find_first_child_by_type(child, 'Div', input_name_to_nodes, recursive=False)
                        if div_node is not None:
                            break
                if div_node is None:
                    continue

                parent_nodes = self.match_parent_path(div_node, ['Sqrt', 'Add', 'ReduceMean', 'Pow', 'Sub'], [1, 0, 0, 0, 0], output_name_to_node)
                if parent_nodes is None:
                    continue

                sqrt_node, second_add_node, reduce_mean_node, pow_node, sub_node = parent_nodes
                if sub_node not in children:
                    continue

                i, add_weight = self.get_constant_input(second_add_node)
                if add_weight is None or add_weight <= 0 or add_weight > 1.0E-5:
                    continue

                if not self.find_constant_input(pow_node, 2.0) == 1:
                    continue

                mul_node = input_name_to_nodes[div_node.output[0]][0]
                if mul_node.op_type != 'Mul':
                    continue

                last_add_node = input_name_to_nodes[mul_node.output[0]][0]
                if last_add_node.op_type != 'Add':
                    continue

                subgraph_nodes = [node]
                subgraph_nodes.extend(children)
                subgraph_nodes.extend([last_add_node, mul_node, div_node, sqrt_node, second_add_node, reduce_mean_node, pow_node])
                if not self.is_safe_to_fuse_nodes(subgraph_nodes, last_add_node.output, input_name_to_nodes, output_name_to_node):
                    continue

                nodes_to_remove.extend(subgraph_nodes)

                weight_input = mul_node.input[1 - self.input_index(div_node.output[0], mul_node)]
                bias_input = last_add_node.input[1 - self.input_index(mul_node.output[0], last_add_node)]
                if parent.op_type == 'Add' and self.is_safe_to_fuse_nodes([parent] + subgraph_nodes, last_add_node.output, input_name_to_nodes, output_name_to_node):
                    nodes_to_remove.append(parent)
                    normalize_node = onnx.helper.make_node(self.normalize_name,
                        inputs=[parent.input[0], parent.input[1], weight_input, bias_input],
                        outputs=[last_add_node.output[0]],
                        name=self.create_node_name(self.normalize_name, name_prefix="SkipLayerNorm"))
                    normalize_node.domain = "com.microsoft"
                    skip_layernorm_nodes.extend([normalize_node])
                else:
                    normalize_node = onnx.helper.make_node('LayerNormalization',
                        inputs=[node.input[0], weight_input, bias_input],
                        outputs=[last_add_node.output[0]])
                    normalize_node.attribute.extend([onnx.helper.make_attribute("epsilon", add_weight)])
                    layernorm_nodes.extend([normalize_node])

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(skip_layernorm_nodes)
        self.add_nodes(layernorm_nodes)
        print("Fused SkipLayerNormalization count:", len(skip_layernorm_nodes))
        print("Fused LayerNormalization count:", len(layernorm_nodes))

    def preprocess(self):
        return

    def optimize(self):
        self. preprocess()
        self.fuse_layer_norm()

        # FastGelu uses approximation for Gelu.  It is faster.
        use_approximation = self.gpu_only
        gelu_op_name = 'FastGelu' if use_approximation else 'Gelu'
        self.fuse_gelu(gelu_op_name)

        self.fuse_reshape()

        self.fuse_attention()
        self.fuse_embed_layer()

        # Fuse Gelu and Add Bias before it.
        self.fuse_add_bias_gelu()
        # Fuse SkipLayerNormalization and Add Bias before it.
        self.fuse_add_bias_skip_layer_norm()

        if self.float16:
            self.convert_model_float32_to_float16()

        self.remove_unused_constant()

        # Use symbolic batch dimension in input and output.
        self.update_dynamic_batch_io()

        print("opset verion", self.model.opset_import[0].version)