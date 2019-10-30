#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# Convert Bert ONNX model exported from PyTorch to use Attention, Gelu, SkipLayerNormalization and
# EmbedLayerNormalization ops to optimize performance on NVidia GPU.

import onnx
import sys
import argparse
import numpy as np
from collections import deque
from onnx import ModelProto, TensorProto, numpy_helper

class OnnxModel:
    def __init__(self, model):
        self.model = model
        self.node_name_counter = {}

    def input_name_to_nodes(self):
        input_name_to_nodes = {}
        for node in self.model.graph.node:
            for input_name in node.input:
                if input_name not in input_name_to_nodes:
                    input_name_to_nodes[input_name] = [node]
                else:
                    input_name_to_nodes[input_name].append(node)
        return input_name_to_nodes

    def output_name_to_node(self):
        output_name_to_node = {}
        for node in self.model.graph.node:
            for output_name in node.output:
                    output_name_to_node[output_name] = node
        return output_name_to_node

    def nodes(self):
        return self.model.graph.node

    def graph(self):
        return self.model.graph

    def remove_node(self, node):
        if node in self.model.graph.node:
            self.model.graph.node.remove(node)

    def remove_nodes(self, nodes_to_remove):
        for node in nodes_to_remove:
            self.remove_node(node)

    def add_node(self, node):
        self.model.graph.node.extend([node])

    def add_nodes(self, nodes_to_add):
        self.model.graph.node.extend(nodes_to_add)

    def add_initializer(self, tensor):
        self.model.graph.initializer.extend([tensor])
 
    def add_input(self, input):
        self.model.graph.input.extend([input])

    @staticmethod
    def replace_node_input(node, old_input_name, new_input_name):
        assert isinstance(old_input_name, str) and isinstance(new_input_name, str)
        for j in range(len(node.input)):
            if node.input[j] == old_input_name:
                node.input[j] = new_input_name

    def replace_input_of_all_nodes(self, old_input_name, new_input_name):
        for node in self.model.graph.node:
            OnnxModel.replace_node_input(node, old_input_name, new_input_name)

    def get_initializer(self,name):
        for tensor in self.model.graph.initializer:
            if tensor.name == name:
                return tensor
        return None

    def get_nodes_by_op_type(self, op_type):
        return [n for n in self.model.graph.node if n.op_type == op_type]

    def get_children(self, node, input_name_to_nodes=None):
        if (input_name_to_nodes is None):
            input_name_to_nodes = self.input_name_to_nodes()
            
        children = []
        for output in node.output:
            if output in input_name_to_nodes:
                for node in input_name_to_nodes[output]:
                    children.append(node)
        return children

    def get_parents(self, node, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        parents = []
        for input in node.input:
            if input in output_name_to_node:
                parents.append(output_name_to_node[input])
        return parents

    def get_parent(self, node, i, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        if len(node.input) <= i:
            return None

        input = node.input[i]
        if input not in output_name_to_node:
            return None

        return output_name_to_node[input]

    def match_parent_path(self, node, parent_op_types, parent_input_index=None, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        if parent_input_index is None:
            parent_input_index = [0] * len(parent_op_types)

        assert(len(parent_input_index) == len(parent_op_types))
        current_node = node
        matched_parents = []
        for i, op_type in enumerate(parent_op_types):
            input_index = parent_input_index[i]
            if input_index >= len(current_node.input):
                return None
            parent = self.get_parent(current_node, input_index, output_name_to_node)
            if parent is None:
                return None
            if parent.op_type == parent_op_types[i]:
                matched_parents.append(parent)
            current_node = parent
        return matched_parents

    def find_first_child_by_type(self, node, child_type, input_name_to_nodes=None, recursive=True):
        children = self.get_children(node, input_name_to_nodes)
        dq = deque(children)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node.op_type == child_type:
                return current_node

            if recursive:
                children = self.get_children(current_node, input_name_to_nodes)
                for child in children:
                    dq.appendleft(child)

        return None

    def find_first_parent_by_type(self, node, parent_type, output_name_to_node=None, recursive=True):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()
            
        parents = self.get_parents(node, output_name_to_node)
        dq = deque(parents)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node.op_type == parent_type:
                return current_node

            if recursive:
                parents = self.get_parents(current_node, output_name_to_node)
                for parent in parents:
                    dq.appendleft(parent)

        return None

    def get_constant_value(self, output_name):
        for node in self.get_nodes_by_op_type('Constant'):
            if node.output[0] == output_name:
                for att in node.attribute:
                    if att.name == 'value':
                        return numpy_helper.to_array(att.t)

    def get_children_subgraph_nodes(self, root_node, stop_nodes, input_name_to_nodes=None):
        if input_name_to_nodes is None:
            input_name_to_nodes = self.input_name_to_nodes()

        children = input_name_to_nodes[root_node.output[0]]

        unique_nodes = []

        dq = deque(children)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node in stop_nodes:
                continue

            if current_node not in unique_nodes:
                unique_nodes.append(current_node)

            for output in current_node.output:
                if output in input_name_to_nodes:
                    children = input_name_to_nodes[output]
                    for child in children:
                        dq.appendleft(child)

        return unique_nodes

    def convert_model_float32_to_float16(self):
        graph = self.model.graph
        initializers = graph.initializer

        for input_value_info in graph.input:
            if input_value_info.type.tensor_type.elem_type == 1:
                input_value_info.type.tensor_type.elem_type = 10

        for output_value_info in graph.output:
            if output_value_info.type.tensor_type.elem_type == 1:
                output_value_info.type.tensor_type.elem_type = 10

        for initializer in initializers:
            if initializer.data_type == 1:
                initializer.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np.float16), initializer.name))

        for node in graph.node:
            if node.op_type == 'Constant':
                for att in node.attribute:
                    if att.name == 'value' and att.t.data_type == 1:
                        att.CopyFrom(onnx.helper.make_attribute("value", numpy_helper.from_array(numpy_helper.to_array(att.t).astype(np.float16))))
            if node.op_type == 'Cast':
                for att in node.attribute:
                    if att.name == 'to' and att.i == 1:
                        att.CopyFrom(onnx.helper.make_attribute("to", 10))
   
    # create a new name for node
    def create_node_name(self, op_type, name_prefix=None):
        if op_type in self.node_name_counter:
            self.node_name_counter[op_type] += 1
        else:
            self.node_name_counter[op_type] = 1

        if name_prefix is not None:
            full_name = name_prefix + str(self.node_name_counter[op_type]) 
        else:
            full_name = op_type + "_" + str(self.node_name_counter[op_type])

        # Check whether the name is taken:
        nodes = self.get_nodes_by_op_type(op_type)
        for node in nodes:
            if node.name == full_name:
                raise Exception("Node name already taken:", full_name)

        return full_name


    def find_graph_input(self, input_name):
        for input in self.model.graph.input:
            if input.name == input_name:
                return input
        return None

    def get_parent_subgraph_nodes(self, node, stop_nodes, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        unique_nodes = []

        parents = self.get_parents(node, output_name_to_node)
        dq = deque(parents)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node in stop_nodes:
                continue

            if current_node not in unique_nodes:
                unique_nodes.append(current_node)

            for input in current_node.input:
                if input in output_name_to_node:
                    dq.appendleft(output_name_to_node[input])

        return unique_nodes

    @staticmethod
    def input_index(node_output, child_node):
        index = 0
        for input in child_node.input:
            if input == node_output:
                return index
            index += 1
        return -1

    def remove_unused_constant(self):
        input_name_to_nodes = self.input_name_to_nodes()

        #remove unused constant
        unused_nodes = []
        nodes = self.nodes()
        for node in nodes:
            if node.op_type == "Constant" and node.output[0] not in input_name_to_nodes:
                unused_nodes.append(node)

        self.remove_nodes(unused_nodes)

        if len(unused_nodes) > 0:
            print("Removed unused constant nodes:", len(unused_nodes))

    def update_graph(self, verbose=False):
        graph = self.model.graph

        remaining_input_names = []
        for node in graph.node:
            if node.op_type != "Constant":
                for input_name in node.input:
                    if input_name not in remaining_input_names:
                        remaining_input_names.append(input_name)
        if verbose:
            print("remaining input names", remaining_input_names)

        # remove graph input that is not used
        inputs_to_remove = []
        for input in graph.input:
            if input.name not in remaining_input_names:
                inputs_to_remove.append(input)
        for input in inputs_to_remove:
            graph.input.remove(input)
        if verbose:
            print("remove unused input ", len(inputs_to_remove), [input.name for input in inputs_to_remove])
        
        # remove weights that are not used
        weights_to_remove = []
        for initializer in graph.initializer:
            if initializer.name not in remaining_input_names:
                weights_to_remove.append(initializer)
        for initializer in weights_to_remove:
            graph.initializer.remove(initializer)
        if verbose:
            print("remove unused initializers:", len(weights_to_remove), [initializer.name for initializer in weights_to_remove])

        self.remove_unused_constant()

class BertOnnxModel(OnnxModel):
    def __init__(self, model, num_heads, hidden_size, sequence_length):
        assert num_heads > 0
        assert hidden_size % num_heads == 0
        assert sequence_length > 0
        
        super(BertOnnxModel, self).__init__(model)
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.mask_input = None
        self.embed_node = None

        # constant node names
        self.normalize_name = "SkipLayerNormalization"
        self.gelu_name = 'FastGelu'
        self.attention_name = 'Attention'

    def get_normalize_nodes(self):
        return self.get_nodes_by_op_type(self.normalize_name)

    def normalize_children_types(self):
        return ['MatMul', 'MatMul', 'MatMul', 'SkipLayerNormalization']

    def set_mask_input(self, input):
        if self.mask_input is not None and input != self.mask_input:
            raise Exception("Different mask inputs", self.mask_input, input)

        self.mask_input = input

    def create_attention_node(self, q_matmul, k_matmul, v_matmul, q_add, k_add, v_add, input, output):
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
            inputs=[input, attention_node_name + '_qkv_weight', attention_node_name + '_qkv_bias', self.mask_input],
            outputs=[output],
            name=attention_node_name)
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([onnx.helper.make_attribute("num_heads", self.num_heads)])

        self.add_node(attention_node)

    def fuse_attention(self, verbose=False):
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []

        for normalize_node in self.get_normalize_nodes():
            # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
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
                continue
            (softmax_qk, add_qk, div_qk, matmul_qk) = qk_nodes

            q_nodes = self.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'MatMul'], [0, 0, 0, 0])
            if q_nodes is None:
                continue
            (transpose_q, reshape_q, add_q, matmul_q) = q_nodes

            k_nodes = self.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0])
            if k_nodes is None:
                continue
            (transpose_k, reshape_k, add_k, matmul_k) = k_nodes

            mask_nodes = self.match_parent_path(add_qk, ['Mul', 'Sub', 'Cast', 'Unsqueeze', 'Unsqueeze'], [1, 0, 1, 0, 0])
            if mask_nodes is None:
                continue
            (mul_mask, sub_mask, cast_mask, unsqueeze_mask, unsqueeze_mask_0) = mask_nodes

            if matmul_v.input[0] == root_input and matmul_q.input[0] == root_input and matmul_v.input[0] == root_input:
                self.set_mask_input(unsqueeze_mask_0.input[0])
                self.create_attention_node(matmul_q, matmul_k, matmul_v, add_q, add_k, add_v, root_input, reshape_qkv.output[0])
                nodes_to_remove.extend([reshape_qkv, transpose_qkv, matmul_qkv])
                nodes_to_remove.extend(qk_nodes)
                nodes_to_remove.extend(q_nodes)
                nodes_to_remove.extend(k_nodes)
                nodes_to_remove.extend(v_nodes)
                nodes_to_remove.extend(mask_nodes)

        self.remove_nodes(nodes_to_remove)
        self.update_graph(verbose)

    def fuse_gelu(self):
        nodes = self.nodes()
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        nodes_to_add = []

        for node in self.get_normalize_nodes():

            children = input_name_to_nodes[node.output[0]]
            if len(children) != 2:
                continue

            children_types = sorted([child.op_type for child in children])
            if children_types != ['MatMul', 'SkipLayerNormalization']:
                continue

            matmul_node = self.find_first_child_by_type(node, 'MatMul', input_name_to_nodes)
            matmul_child = input_name_to_nodes[matmul_node.output[0]]
            if len(matmul_child) != 1 or matmul_child[0].op_type != 'Add':
                continue
            add_node = matmul_child[0]

            children = input_name_to_nodes[add_node.output[0]]

            children_types = sorted([child.op_type for child in children])
            if children_types != ['Div', 'Mul']:
                continue

            matmul_2 = self.find_first_child_by_type(add_node, 'MatMul', input_name_to_nodes)
            if matmul_2 is None:
                continue

            subgraph_nodes = self.get_children_subgraph_nodes(add_node, [matmul_2], input_name_to_nodes)
            if len(subgraph_nodes) != 5:
                continue

            nodes_to_remove.append(add_node)
            nodes_to_remove.extend(subgraph_nodes)
            bias_input = add_node.input[1] if (add_node.input[0] == matmul_node.output[0]) else add_node.input[0]
            gelu_node = onnx.helper.make_node(self.gelu_name,
                inputs=[matmul_node.output[0], bias_input],
                outputs=[matmul_2.input[0]])
            gelu_node.domain = "com.microsoft"
            nodes_to_add.append(gelu_node)

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)

    def fuse_reshape(self):
        nodes = self.nodes()
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        nodes_to_add = []

        for reshape_node in self.get_nodes_by_op_type('Reshape'):
            concat_node = output_name_to_node[reshape_node.input[1]]
            if concat_node.op_type != 'Concat' or len(concat_node.input) < 3:
                continue

            path = self.match_parent_path(concat_node, ['Unsqueeze', 'Gather', 'Shape'], [0, 0, 0], output_name_to_node)
            if path is None:
                continue
            (unsqueeze_0, gather_0, shape_0) = path

            path = self.match_parent_path(concat_node, ['Unsqueeze', 'Gather', 'Shape'], [1, 0, 0], output_name_to_node)
            if path is None:
                continue
            (unsqueeze_1, gather_1, shape_1) = path

            shape = []
            gather_value = self.get_constant_value(gather_0.input[1])
            if gather_value == 0:
                shape.append(0)

            gather_value = self.get_constant_value(gather_1.input[1])
            if gather_value == 1:
                shape.append(0)

            if len(shape) != 2:
                continue

            if (len(concat_node.input) > 2):
                concat_2 = self.get_initializer(concat_node.input[2])
                if concat_2 is None:
                    continue
                shape.extend(numpy_helper.to_array(concat_2))

            if (len(concat_node.input) > 3):
                concat_3 = self.get_initializer(concat_node.input[3])
                if concat_3 is None:
                    continue
                shape.extend(numpy_helper.to_array(concat_3))
            shape_value = np.asarray(shape, dtype=np.int64)

            constant_shape_name = self.create_node_name('Constant', 'constant_shape')
            new_node = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[constant_shape_name],
                value=onnx.helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT64,
                    dims=shape_value.shape,
                    vals=shape_value))
            reshape_node.input[1] = constant_shape_name
            nodes_to_remove.extend([concat_node, unsqueeze_0, unsqueeze_1, gather_0, gather_1, shape_0, shape_1])
            nodes_to_add.append(new_node)

        print("Fused reshape count:", len(nodes_to_add))

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

      Optional graph is used to generate position list (0, 1, ...). It can be a constant in some model.
    """
    def fuse_embed_layer(self, verbose=False):
        if self.mask_input is None:
            print("skip embed layer fusion since mask input is not found")
            return

        nodes = self.nodes()
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()
        mask_input_name = self.mask_input

        nodes_to_remove = []
        nodes_to_add = []

        # Find the first normalize node could be embedding layer.
        normalize_node = None
        for node in self.get_normalize_nodes():
            if self.match_parent_path(node, ['Add', 'Gather'], [0, 0]) is not None:
                if self.find_first_child_by_type(node, 'Attention', input_name_to_nodes, recursive=False) is not None:
                    normalize_node = node
                    break

        if normalize_node is None:
            print("did not find embedding layer")

        # Here we assume the order of embedding is word_embedding + position_embedding + segment_embedding.
        word_embedding_path = self.match_parent_path(normalize_node, ['Add', 'Gather'], [0, 0])
        if word_embedding_path is None:
            print("Failed to find word embedding")
            return
        add_node, word_embedding_gather = word_embedding_path

        position_embedding_path = self.match_parent_path(add_node, ['Gather', 'Expand', 'Shape'], [1, 1, 1])
        if position_embedding_path is None:
            print("Failed to find position embedding")
            return
        position_embedding_gather, position_embedding_expand, position_embedding_shape = position_embedding_path

        segment_embedding_path = self.match_parent_path(normalize_node, ['Gather'], [1])
        if segment_embedding_path is None:
            print("failed to find segment embedding")
            return
        segment_embedding_gather = segment_embedding_path[0]

        input_ids = word_embedding_gather.input[1]
        segment_ids = segment_embedding_gather.input[1]

        if position_embedding_shape.input[0] != input_ids:
            print("position and word embedding is expected to be applied on same input")
            return

        subgraph_nodes = self.get_parent_subgraph_nodes(position_embedding_expand, [input_ids], output_name_to_node)

        nodes_to_remove.extend(subgraph_nodes)
        nodes_to_remove.extend([normalize_node, add_node, segment_embedding_gather, word_embedding_gather, position_embedding_gather, position_embedding_expand])

        embed_node = onnx.helper.make_node('EmbedLayerNormalization',
                        inputs=[input_ids, segment_ids, mask_input_name, 
                                word_embedding_gather.input[0], position_embedding_gather.input[0], segment_embedding_gather.input[0],
                                normalize_node.input[2], normalize_node.input[3]], # gamma and beta
                        outputs=["embed_output", "mask_idx"],
                        name="EmbedLayer")
        embed_node.domain = "com.microsoft"
        # store embed node for other processing
        self.embed_node = embed_node

        nodes_to_add.extend([embed_node])

        self.replace_input_of_all_nodes(normalize_node.output[0], 'embed_output')
        self.replace_input_of_all_nodes(mask_input_name, 'mask_idx')

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)
        self.update_graph(verbose)

    def get_batch_size_from_graph_input(self):
        graph = self.graph()
        for input in graph.input:
            if input.name in self.embed_node.input[:3]:
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
        for input in graph.input:
            if input.name in self.embed_node.input[:3]: # Only the first 3 inputs of embed node need int32 conversion.
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

    def cast_input_to_int32(self):
        for input in self.embed_node.input[:3]:
            graph_input = self.find_graph_input(input)
            if graph_input is not None and graph_input.type.tensor_type.elem_type == TensorProto.INT64:
                cast_output = input + '_int32'
                cast_node = onnx.helper.make_node('Cast', inputs=[input], outputs=[cast_output])
                cast_node.attribute.extend([onnx.helper.make_attribute("to", int(TensorProto.INT32))])
                self.replace_input_of_all_nodes(input, cast_output)
                self.add_node(cast_node)

    # Update input and output using dynamic batch
    def update_dynamic_batch_io(self, dynamic_batch_dim='batch'):
        dynamic_batch_inputs = {}
        for input in self.model.graph.input:
            for embed_input in self.embed_node.input[:3]:
                if embed_input == input.name:
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
                                 |                                               ^
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
    """
    def fuse_layer_norm(self):
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        nodes_to_add = []

        for node in self.nodes():
            if node.op_type == 'Add':
                children = self.get_children(node, input_name_to_nodes)
                children_types = sorted([child.op_type for child in children])
                if children_types != ["ReduceMean", "Sub"] and children_types != ["ReduceMean", "Sub", "Sub"]:
                    continue

                div_node = None
                for child in children:
                        if child.op_type == 'Sub':
                            div_node = self.find_first_child_by_type(child, 'Div', input_name_to_nodes, recursive=False)
                            if div_node is not None:
                                break
                if div_node is None:
                    continue

                parent_nodes = self.match_parent_path(div_node, ['Sqrt', 'Add', 'ReduceMean', 'Pow', 'Sub', 'Add'], [1, 0, 0, 0, 0, 0], output_name_to_node)
                if parent_nodes is None:
                    continue

                sqrt_node, second_add_node, reduce_mean_node, pow_node, sub_node, first_add_node = parent_nodes
                if first_add_node != node:
                    continue

                mul_node = input_name_to_nodes[div_node.output[0]][0]
                if mul_node.op_type != 'Mul':
                    continue

                last_add_node = input_name_to_nodes[mul_node.output[0]][0]
                if last_add_node.op_type != 'Add':
                    continue

                nodes_to_remove.append(node)
                nodes_to_remove.extend(children)
                nodes_to_remove.extend([last_add_node, mul_node, div_node, sqrt_node, second_add_node, reduce_mean_node, pow_node])

                normalize_node_name = self.create_node_name(self.normalize_name, name_prefix="SkipLayerNorm")
                inputs = [i for i in node.input]
                inputs.extend([mul_node.input[0], last_add_node.input[1]])
                normalize_node = onnx.helper.make_node(self.normalize_name,
                    inputs=inputs,
                    outputs=[last_add_node.output[0]],
                    name=normalize_node_name)
                normalize_node.domain = "com.microsoft"
                nodes_to_add.extend([normalize_node])

        self.remove_nodes(nodes_to_remove)
        self.add_nodes(nodes_to_add)
        print("Fused layer normalization count:", len(nodes_to_add))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)

    # model parameters
    parser.add_argument('--num_heads', required=False, type=int, default=12, help="number of attention heads")
    parser.add_argument('--hidden_size', required=False, type=int, default=768)
    parser.add_argument('--sequence_length', required=False, type=int, default=128)

    # Use int32 (instead of int64) tensor as input to avoid unnecessary data type cast.
    parser.add_argument('--input_int32', required=False, action='store_true')
    parser.set_defaults(input_int32=False)

    # For NVidia GPU with Tensor Core like V100 and T4, half-precision float brings better performance.
    parser.add_argument('--float16', required=False, action='store_true')
    parser.set_defaults(float16=False)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    model = ModelProto()
    with open(args.input, "rb") as f:
        model.ParseFromString(f.read())

    bert_model = BertOnnxModel(model, args.num_heads, args.hidden_size, args.sequence_length)

    bert_model.fuse_layer_norm()

    bert_model.fuse_gelu()

    bert_model.fuse_reshape()

    bert_model.fuse_attention(args.verbose)

    bert_model.fuse_embed_layer(args.verbose)
    
    if bert_model.embed_node is None:
        print("Failed to fuse embedding layer.")
        return

    if args.input_int32:
        bert_model.change_input_to_int32()
    else:
        bert_model.cast_input_to_int32()

    if args.float16:
        bert_model.convert_model_float32_to_float16()

    with open(args.output, "wb") as out:
        out.write(bert_model.model.SerializeToString())

main()
