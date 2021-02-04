#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
import numpy as np
from logging import getLogger
from enum import Enum
from onnx import helper, numpy_helper, TensorProto
from onnx_model import OnnxModel
from fusion_base import Fusion
from fusion_utils import FusionUtils

logger = getLogger(__name__)


class AttentionMaskFormat:
    MaskIndexEnd = 0
    MaskIndexEndAndStart = 1
    AttentionMask = 2
    NoMask = 3


class AttentionMask():
    """
    Fuse Attention subgraph into one Attention node.
    """
    def __init__(self, model: OnnxModel):
        self.model = model
        # A lookup table with mask input as key, and mask index output as value
        self.mask_indice = {}
        # A lookup table with mask input as key, and cast (to int32) output as value
        self.mask_casted = {}
        self.utils = FusionUtils(model)
        self.mask_format = AttentionMaskFormat.MaskIndexEnd

    def set_mask_format(self, mask_format: AttentionMaskFormat):
        self.mask_format = mask_format

    def set_mask_indice(self, mask, mask_index):
        if mask in self.mask_indice:
            assert mask_index == self.mask_indice[mask]
        self.mask_indice[mask] = mask_index

    def get_first_mask(self):
        assert len(self.mask_indice) > 0
        return next(iter(self.mask_indice))

    def process_mask(self, input):
        if self.mask_format == AttentionMaskFormat.NoMask:
            return None

        if input in self.mask_indice:
            return self.mask_indice[input]

        # Add cast to convert int64 to int32
        if self.model.find_graph_input(input):
            casted, input_name = self.utils.cast_graph_input_to_int32(input)
        else:
            input_name, cast_node = self.utils.cast_input_to_int32(input)
            casted = True

        if casted:
            self.mask_casted[input] = input_name

        # Attention supports int32 attention mask (2D) since 1.4.0
        if self.mask_format == AttentionMaskFormat.AttentionMask:
            self.mask_indice[input] = input_name
            return input_name

        # Add a mask processing node to convert attention mask to mask index (1D)
        output_name = self.model.create_node_name('mask_index')
        mask_index_node = helper.make_node('ReduceSum',
                                           inputs=[input_name],
                                           outputs=[output_name],
                                           name=self.model.create_node_name('ReduceSum', 'MaskReduceSum'))
        mask_index_node.attribute.extend([helper.make_attribute("axes", [1]), helper.make_attribute("keepdims", 0)])
        self.model.add_node(mask_index_node)

        self.mask_indice[input] = output_name
        return output_name


class FusionAttention(Fusion):
    """
    Fuse Attention subgraph into one Attention node.
    """
    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int, attention_mask: AttentionMask):
        super().__init__(model, "Attention", ["SkipLayerNormalization", "LayerNormalization"])
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_mask = attention_mask

    def create_attention_node(self, mask_index, q_matmul, k_matmul, v_matmul, q_add, k_add, v_add, input, output):
        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])
        q_bias = self.model.get_initializer(q_add.input[1])
        k_bias = self.model.get_initializer(k_add.input[1])
        v_bias = self.model.get_initializer(v_add.input[1])

        if q_weight is None:
            print(f"{q_matmul.input[1]} is not initializer. Please set do_constant_folding=True in torch.onnx.export")
            return None
        if not (k_weight and v_weight and q_bias and k_bias):
            return None

        qw = numpy_helper.to_array(q_weight)
        assert qw.shape == (self.hidden_size, self.hidden_size)

        kw = numpy_helper.to_array(k_weight)
        assert kw.shape == (self.hidden_size, self.hidden_size)

        vw = numpy_helper.to_array(v_weight)
        assert vw.shape == (self.hidden_size, self.hidden_size)

        qkv_weight = np.stack((qw, kw, vw), axis=-2)

        qb = numpy_helper.to_array(q_bias)
        assert qb.shape == (self.hidden_size, )

        kb = numpy_helper.to_array(k_bias)
        assert kb.shape == (self.hidden_size, )

        vb = numpy_helper.to_array(v_bias)
        assert vb.shape == (self.hidden_size, )

        qkv_bias = np.stack((qb, kb, vb), axis=-2)

        attention_node_name = self.model.create_node_name('Attention')

        weight = helper.make_tensor(name=attention_node_name + '_qkv_weight',
                                    data_type=TensorProto.FLOAT,
                                    dims=[self.hidden_size, 3 * self.hidden_size],
                                    vals=qkv_weight.flatten().tolist())
        # Sometimes weights and bias are stored in fp16
        if q_weight.data_type == 10:
            weight.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weight).astype(np.float16), weight.name))
        self.model.add_initializer(weight)

        bias = helper.make_tensor(name=attention_node_name + '_qkv_bias',
                                  data_type=TensorProto.FLOAT,
                                  dims=[3 * self.hidden_size],
                                  vals=qkv_bias.flatten().tolist())
        if q_bias.data_type == 10:
            bias.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(bias).astype(np.float16), bias.name))
        self.model.add_initializer(bias)

        attention_inputs = [input, attention_node_name + '_qkv_weight', attention_node_name + '_qkv_bias']
        if mask_index is not None:
            attention_inputs.append(mask_index)

        attention_node = helper.make_node('Attention',
                                          inputs=attention_inputs,
                                          outputs=[output],
                                          name=attention_node_name)
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", self.num_heads)])

        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node
        if normalize_node.op_type == 'LayerNormalization':
            add_before_layernorm = self.model.match_parent(normalize_node, 'Add', 0)
            if add_before_layernorm is not None:
                start_node = add_before_layernorm
            else:
                return

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(start_node, ['Add', 'MatMul', 'Reshape', 'Transpose', 'MatMul'],
                                                 [None, 0, 0, 0, 0])
        einsum_node = None
        if qkv_nodes is not None:
            (_, matmul_qkv, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
        else:
            # Match Albert
            qkv_nodes = self.model.match_parent_path(start_node, ['Add', 'Einsum', 'Transpose', 'MatMul'], [1, 0, 0, 0])
            if qkv_nodes is not None:
                (_, einsum_node, transpose_qkv, matmul_qkv) = qkv_nodes
            else:
                return

        other_inputs = []
        for i, input in enumerate(start_node.input):
            if input not in output_name_to_node:
                continue

            if input == qkv_nodes[0].output[0]:
                continue
            other_inputs.append(input)
        if len(other_inputs) != 1:
            return

        root_input = other_inputs[0]
        """
        Match flaubert                     Mask
                                            |
        Mul --> LayerNormalization -->  Attention --> MatMul --> Add
         |                                                        |
         |                                                        |
         +---------------------------------------------------------
        """
        mul_before_layernorm = self.model.match_parent(start_node, 'Mul', 0)
        if mul_before_layernorm is not None:
            mul_children = input_name_to_nodes[mul_before_layernorm.output[0]]
            if mul_children is not None and len(mul_children) == 2:
                layernorm_node = mul_children[1]
                if layernorm_node.op_type == 'LayerNormalization':
                    root_input = layernorm_node.output[0]
                else:
                    return
            else:
                return

        children = input_name_to_nodes[root_input]
        children_types = [child.op_type for child in children]
        if children_types.count('MatMul') != 3:
            return

        v_nodes = self.model.match_parent_path(matmul_qkv, ['Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (_, _, add_v, matmul_v) = v_nodes

        is_distill = False
        qk_nodes = self.model.match_parent_path(matmul_qkv, ['Softmax', 'Add', 'Div', 'MatMul'], [0, 0, 0, 0])
        if qk_nodes is None:
            qk_nodes = self.model.match_parent_path(matmul_qkv, ['Softmax', 'Add', 'Mul', 'MatMul'], [0, 0, 0, 0])
            if qk_nodes is None:
                qk_nodes = self.model.match_parent_path(matmul_qkv, ['Softmax', 'Where', 'MatMul', 'Div'], [0, 0, 2, 0])
                is_distill = True
                if qk_nodes is None:
                    logger.debug("fuse_attention: failed to match qk path")
                    return

        add_qk = None
        matmul_qk = None
        where_qk = None
        if is_distill:
            (_, where_qk, matmul_qk, _) = qk_nodes
        else:
            (_, add_qk, _, matmul_qk) = qk_nodes

        q_nodes = self.model.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'MatMul'], [0, 0, 0, 0])
        if q_nodes is None:
            q_nodes = self.model.match_parent_path(matmul_qk, ['Div', 'Transpose', 'Reshape', 'Add', 'MatMul'],
                                                   [0, 0, 0, 0, 0])
            if q_nodes is None:
                logger.debug("fuse_attention: failed to match q path")
                return
        add_q = q_nodes[-2]
        matmul_q = q_nodes[-1]

        k_nodes = self.model.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0])
        if k_nodes is None:
            k_nodes = self.model.match_parent_path(matmul_qk, ['Transpose', 'Transpose', 'Reshape', 'Add', 'MatMul'],
                                                   [1, 0, 0, 0, 0])
            if k_nodes is None:
                logger.debug("fuse_attention: failed to match k path")
                return
        add_k = k_nodes[-2]
        matmul_k = k_nodes[-1]

        # Note that Cast might be removed by OnnxRuntime so we match two patterns here.
        mask_nodes = None
        if is_distill:
            _, mask_nodes, _ = self.model.match_parent_paths(where_qk,
                                                             [(['Expand', 'Reshape', 'Equal'], [0, 0, 0]),
                                                              (['Cast', 'Expand', 'Reshape', 'Equal'], [0, 0, 0, 0])],
                                                             output_name_to_node)
        else:
            _, mask_nodes, _ = self.model.match_parent_paths(
                add_qk, [(['Mul', 'Sub', 'Cast', 'Unsqueeze', 'Unsqueeze'], [1, 0, 1, 0, 0]),
                         (['Mul', 'Sub', 'Unsqueeze', 'Unsqueeze'], [1, 0, 1, 0])], output_name_to_node)
        if mask_nodes is None:
            logger.debug("fuse_attention: failed to match mask path")
            return

        if matmul_v.input[0] == root_input and matmul_q.input[0] == root_input and matmul_v.input[0] == root_input:
            mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])

            attention_last_node = reshape_qkv if einsum_node is None else transpose_qkv

            new_node = self.create_attention_node(mask_index, matmul_q, matmul_k, matmul_v, add_q, add_k, add_v,
                                                  root_input, attention_last_node.output[0])
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)

            if einsum_node is not None:
                unique_index = einsum_node.input[0]
                new_edge = "edge_modified_" + unique_index
                shape_tensor = helper.make_tensor(name="shape_modified_tensor" + unique_index,
                                                  data_type=TensorProto.INT64,
                                                  dims=[4],
                                                  vals=np.int64(
                                                      [0, 0, self.num_heads,
                                                       int(self.hidden_size / self.num_heads)]).tobytes(),
                                                  raw=True)
                self.model.add_initializer(shape_tensor)
                self.model.add_node(
                    helper.make_node("Reshape", [attention_last_node.output[0], shape_tensor.name], [new_edge],
                                     "reshape_modified_" + unique_index))
                einsum_node.input[0] = new_edge

            self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)
            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes)

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            #self.nodes_to_remove.extend(mask_nodes)
            self.prune_graph = True
