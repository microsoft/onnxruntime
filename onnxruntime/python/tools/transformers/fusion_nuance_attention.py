# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from fusion_attention import AttentionMask, FusionAttention
from onnx import TensorProto, helper, numpy_helper, NodeProto
from onnx_model import OnnxModel
import numpy as np
from fusion_utils import FusionUtils
from typing import List, Optional, Tuple, Union
import sys
import random

logger = logging.getLogger(__name__)


class FusionNuanceAttention(FusionAttention):
    """
    Fuse Bart Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
        self.casted_attention_mask = {}
        self.utils = FusionUtils(model)
        super().__init__(model, hidden_size, num_heads, attention_mask)

    def check_runtime_shape_path(
        self,
        reshape_qkv_2,
        reshape_qkv_1,
        reshape_q_2,
        reshape_k_2,
        reshape_v_2,
        root_input,
    ):
        concat_qkv_2_path = self.model.match_parent_path(reshape_qkv_2, ["Concat"], [1])
        if concat_qkv_2_path is None:
            return False
        concat_qkv_2 = concat_qkv_2_path[0]

        reshape_qkv_2_path_1 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [0, 0, 0])
        reshape_qkv_2_path_2 = self.model.match_parent_path(concat_qkv_2, ["Unsqueeze", "Gather", "Shape"], [1, 0, 0])
        if reshape_qkv_2_path_1 is None or reshape_qkv_2_path_2 is None:
            return False

        _, gather_1, shape_1 = reshape_qkv_2_path_1
        _, gather_2, shape_2 = reshape_qkv_2_path_2

        if shape_1.input[0] != root_input or shape_2.input[0] != root_input:
            return False

        reshape_qkv_1_path_1 = self.model.match_parent_path(reshape_qkv_1, ["Concat", "Unsqueeze", "Gather"], [1, 0, 0])
        reshape_qkv_1_path_2 = self.model.match_parent_path(reshape_qkv_1, ["Concat", "Unsqueeze", "Gather"], [1, 2, 0])
        if reshape_qkv_1_path_1 is None or reshape_qkv_1_path_2 is None:
            return False
        if reshape_qkv_1_path_1[-1].name != gather_1.name or reshape_qkv_1_path_2[-1].name != gather_2.name:
            return False

        reshape_q_2_path = self.model.match_parent_path(reshape_q_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        reshape_k_2_path = self.model.match_parent_path(reshape_k_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        reshape_v_2_path = self.model.match_parent_path(reshape_v_2, ["Concat", "Unsqueeze", "Mul"], [1, 0, 0])
        if reshape_q_2_path is None or reshape_k_2_path is None or reshape_v_2_path is None:
            return False

        mul_q = reshape_q_2_path[-1]
        mul_k = reshape_k_2_path[-1]
        mul_v = reshape_v_2_path[-1]

        gather_1_out = gather_1.output[0]
        if mul_q.input[0] != gather_1_out or mul_k.input[0] != gather_1_out or mul_v.input[0] != gather_1_out:
            return False

        return True

    def reshape_kv(self, past_k: str, past_v: str) -> (str, str):
        """Reshape past_k and past_v from 4D to 3D to use as inputs for multihead attention node.

        Args:
            past_k (str): name of past K value of shape 4D
            past_v (str): name of past V value of shape 4D

        Returns:
            k_3d (str): name of past K value of shape 3D
            v_3d (str): name of past V value of shape 3D
        """
        # Reshape past_k and past_v from (B,N,P,H) to (B,P,N*H)
        # B = batch size, N = num heads, P = past seq len, H = head size

        # Create initializer for reshaping past_k and past_v
        new_dims_name = "kv_5d_to_4d"
        new_dims = self.model.get_initializer(new_dims_name)
        if new_dims is None:
            new_dims = numpy_helper.from_array(
                np.array([16, 8, 72, -1], dtype="int64"), name=new_dims_name
            )
            self.model.add_initializer(new_dims, self.this_graph_name)

        reshape_k_name = self.model.create_node_name("Reshape")
        reshape_v_name = self.model.create_node_name("Reshape")
        k_4d_name = (past_k + "_4d").replace(".", "_")
        v_4d_name = (past_v + "_4d").replace(".", "_")

        k_4d = helper.make_node(
            "Reshape",
            inputs=[past_k, new_dims_name],
            outputs=[k_4d_name],
            name=reshape_k_name,
        )
        v_4d = helper.make_node(
            "Reshape",
            inputs=[past_v, new_dims_name],
            outputs=[v_4d_name],
            name=reshape_v_name,
        )

        # Add reshape nodes to graph
        self.nodes_to_add.append(k_4d)
        self.nodes_to_add.append(v_4d)
        self.node_name_to_graph_name[reshape_k_name] = self.this_graph_name
        self.node_name_to_graph_name[reshape_v_name] = self.this_graph_name

        return k_4d_name, v_4d_name

    def rel_pos_bias_node(self, skip_name: str, reshape_name: str, matmul_parent: str, unsqueeze_0_parent: str, unsqueeze_1_parent: str,
    unsqueeze_2_parent: str):
        """Split kv_node containing present KV values into separate present K and present V values.

        Args:
            present_k_name (str): name of output to store present K value in
            present_v_name (str): name of output to store present V value in
            kv_node (str): name of present KV values
        """
        # Split kv_node into present_k and present_v nodes

        # Create initializers for indexing kv_node, whose shape is (24,B,N,P,H)

        # Create nodes to index kv_node
        rel_bias_name = self.model.create_node_name("Concat")
        rel_bias_node = helper.make_node(
            "Concat",
            inputs=[skip_name, matmul_parent, unsqueeze_0_parent, unsqueeze_1_parent, unsqueeze_2_parent],
            outputs=[reshape_name],
            name=rel_bias_name,
            axis=0,
        )

        # Add gather nodes to graph
        self.nodes_to_add.append(rel_bias_node)
        self.node_name_to_graph_name[rel_bias_name] = self.this_graph_name

        return rel_bias_node

    def cast_attention_mask(self, input_name):
        if input_name in self.casted_attention_mask:
            attention_mask_input_name = self.casted_attention_mask[input_name]
        elif self.model.find_graph_input(input_name):
            casted, attention_mask_input_name = self.utils.cast_graph_input_to_int32(input_name)
            self.casted_attention_mask[input_name] = attention_mask_input_name
        else:
            attention_mask_input_name, cast_node = self.utils.cast_input_to_int32(input_name)
            self.casted_attention_mask[input_name] = attention_mask_input_name
        return attention_mask_input_name

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [None, 1, 0, 0, 0],
        )
        if qkv_nodes is not None:
            (add_qkv, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
        else:
            return

        other_inputs = []
        for input in normalize_node.input:
            if input not in output_name_to_node:
                continue
            if input == qkv_nodes[0].output[0]:
                continue
            other_inputs.append(input)
        if len(other_inputs) != 1:
            return
        root_input = other_inputs[0]

        # Sometimes the input name to the attention MatMul nodes does not match the input name to the end
        # SkipLayerNormalization node (name saved in root_input). We find the true input name to the MatMul
        # nodes by getting the initial SkipLayerNormalization node and checking how many MatMul nodes are
        # children nodes for each of its output names.
        """
                                        root_input
                    +---------------------------------------------------+
                    |                                                   |
                    |                                                   |
        SkipLayerNormalization --> Attention --> MatMul --> SkipLayerNormalization
        """
        skip_layernorm = output_name_to_node[root_input]
        # For some attention blocks, the end SkipLayerNormalization node may point to an Add node whose
        # child is the LayerNormalization node.
        if skip_layernorm.op_type == "Add":
            skip_layernorm = self.model.get_children(skip_layernorm)[0]
        for output in skip_layernorm.output:
            if not output:
                continue
            children = input_name_to_nodes[output]
            children_types = [child.op_type for child in children]
            if children_types.count("MatMul") >= 1:
                root_input = output
                break

        graph_input_names = set([node.name for node in self.model.graph().input])
        graph_output_names = set([node.name for node in self.model.graph().output])

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Concat", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 1, 0, 0, 0],
        )
        v_nodes_with_past_self_attn = self.model.match_parent_path(
            # Decoder attention with past value concatenated before MatMul
            matmul_qkv,
            ["Concat", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 1, 0, 0, 1],
        )
        v_nodes_with_past_cross_attn = self.model.match_parent_path(
            # Decoder attention with past value directly used in MatMul
            matmul_qkv,
            ["Reshape"],
            [1],
        )
        past_v, present_v = "", ""
        reshape_v_2, add_v = None, None
        if v_nodes is not None:
            (_, transpose_v, reshape_v, add_v, matmul_v) = v_nodes
            # For initial pass through encoder-decoder_with_past to get starting past values (beam search)
            transpose_children = self.model.get_children(transpose_v, None)
            concat_children = self.model.get_children(transpose_children[0], None)
            gather_children = self.model.get_children(concat_children[1], None)
            present_v = gather_children[0]
        elif v_nodes_with_past_self_attn is not None:
            (concat_v, transpose_v, reshape_v, add_v, matmul_v) = v_nodes_with_past_self_attn
            v_nodes = v_nodes_with_past_self_attn
            concat_children = self.model.get_children(concat_v, None)
            gather_children = self.model.get_children(concat_children[1], None)
            concat_parent = self.model.get_parent(concat_v, 0, None)
            present_v = concat_children[1].input[0]
            past_v = concat_parent.output[0]
        elif (
            v_nodes_with_past_cross_attn is not None and v_nodes_with_past_cross_attn[-1].input[0] in graph_input_names
        ):
            v_nodes = v_nodes_with_past_cross_attn
            past_v = v_nodes[-1].input[0]
            present_v = v_nodes[-1].output[0]
            if present_v not in graph_output_names:
                identity_node_v = list(
                    filter(lambda node: node.op_type == "Identity", self.model.input_name_to_nodes()[past_v])
                )
                present_v = identity_node_v[0].output[0] if len(identity_node_v) == 1 else ""
        else:
            logger.debug("fuse_attention: failed to match v path")
            return

        input_mask_nodes = None
        qk_nodes = self.model.match_parent_path(
            matmul_qkv, ["Softmax", "Add", "Add", "MatMul"], [0, 0, 0, 0]
        )
        if qk_nodes is not None:
            (_, mask_add, add_qk, matmul_qk) = qk_nodes
            input_mask_nodes = self.model.match_parent_path(
                mask_add,
                [
                    "Unsqueeze", "Unsqueeze", "Where"
                ],
                [1, 0, 0],
            )  # yapf: disable
            if input_mask_nodes is None:
                logger.debug("fuse_attention: failed to match unidirectional mask path")
                return

        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Mul", "Transpose", "Reshape", "Add", "MatMul"],
            [0, 0, 0, 0, None],
        )
        if q_nodes is not None:
            (mul_q, transpose_q, reshape_q, add_q, matmul_q) = q_nodes
        else:
            return

        k_nodes_with_bias = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 0, 0, 0, 1],
        )
        k_nodes_no_bias = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Reshape", "Transpose", "Reshape", "MatMul"],
            [1, 0, 0, 0, 0],
        )
        k_nodes_no_bias_with_past_self_attn = self.model.match_parent_path(
            # Decoder attention with past key concatenated before MatMul
            matmul_qk,
            ["Transpose", "Concat", "Reshape", "Add", "MatMul"],
            [1, 0, 1, 0, None],
        )
        k_nodes_no_bias_with_past_cross_attn = self.model.match_parent_path(
            # Decoder attention with past key directly used in MatMul
            matmul_qk,
            ["Transpose", "Reshape"],
            [1, 0],
        )
        past_k, present_k = "", ""
        reshape_k_2, reshape_k_1, matmul_k = None, None, None
        if k_nodes_with_bias is not None:
            _, reshape_k_2, transpose_k_1, reshape_k_1, add_k, matmul_k = k_nodes_with_bias
            k_nodes = k_nodes_with_bias
        elif k_nodes_no_bias is not None:
            _, reshape_k_2, transpose_k_1, reshape_k_1, matmul_k = k_nodes_no_bias
            k_nodes = k_nodes_no_bias
            # For initial pass through encoder-decoder_with_past to get starting past values (beam search)
            present_k = transpose_k_1.output[0]
        elif k_nodes_no_bias_with_past_self_attn is not None:
            (_, concat_k, reshape_k, add_k, matmul_k) = k_nodes_no_bias_with_past_self_attn
            k_nodes = k_nodes_no_bias_with_past_self_attn
            concat_parent = self.model.get_parent(concat_k, 0, None)
            transpose_parent = self.model.get_parent(concat_parent, 0, None)
            concat_children = self.model.get_children(concat_k, None)
            transpose_children = self.model.get_children(concat_children[1], None)
            gather_children = self.model.get_children(transpose_children[0], None)
            past_k = transpose_parent.output[0]
            present_k = concat_children[1].output[0]
        elif (
            k_nodes_no_bias_with_past_cross_attn is not None
            and k_nodes_no_bias_with_past_cross_attn[-1].input[0] in graph_input_names
        ):
            k_nodes = k_nodes_no_bias_with_past_cross_attn
            past_k = k_nodes[-1].input[0]
            present_k = k_nodes[-1].output[0]
            if present_k not in graph_output_names:
                identity_node_k = list(
                    filter(lambda node: node.op_type == "Identity", self.model.input_name_to_nodes()[past_k])
                )
                present_k = identity_node_k[0].output[0] if len(identity_node_k) == 1 else ""
        else:
            return

        three_root_inputs = past_k and past_v and matmul_k is None and "matmul_v" not in locals()

        one_root_input = (
            not three_root_inputs
            and matmul_k.input[0] == root_input
            and matmul_q.input[0] == root_input
            and matmul_v.input[0] == root_input
        )
        two_root_inputs = (
            not three_root_inputs
            and matmul_q.input[0] == root_input
            and matmul_k.input[0] == matmul_v.input[0]
            and matmul_k.input[0] != matmul_q.input[0]
        )

        # There are 5 types of attention:
        # 1) Encoder attention with one_root_input=True and qk_nodes=qk_nodes_1
        # 2) Decoder attention with one_root_input=True and qk_nodes=qk_nodes_2
        # 3) Decoder attention with past with one_root_input=True and qk_nodes=qk_nodes_1 and past_k=past_decoder_key and past_v=past_decoder_value
        # 4) Decoder cross attention with two_root_inputs=True and qk_nodes=qk_nodes_1
        # 5) Decoder cross attention with past with three_root_inputs=True and qk_nodes=qk_nodes_1
        encoder_attention = one_root_input and qk_nodes
        decoder_attention = one_root_input and qk_nodes
        decoder_attention_with_past = encoder_attention and past_k and past_v
        decoder_cross_attention = two_root_inputs and qk_nodes == qk_nodes_1
        decoder_cross_attention_with_past = three_root_inputs and qk_nodes == qk_nodes_1

        attention_mask_input_name = ""
        new_add = ""
        if input_mask_nodes is not None:
            input_name = input_mask_nodes[2].output[0]
            attention_mask_input_name = self.cast_attention_mask(input_name)
            constant_tensor = helper.make_tensor(
                name="Constant_mask",
                data_type=TensorProto.INT32,
                dims=[1],
                vals=[-(2147483648)],
            )
            new_add = "Add_cast"
            add_name = self.model.create_node_name("Add")
            self.model.add_initializer(constant_tensor, self.this_graph_name)
            self.model.add_node(
                helper.make_node(
                    "Add",
                    [attention_mask_input_name, constant_tensor.name],
                    [new_add],
                    add_name,
                ),
                self.this_graph_name,
            )

        if (
            encoder_attention
            or decoder_attention
            or decoder_attention_with_past
            or decoder_cross_attention
            or decoder_cross_attention_with_past
        ):
            attention_last_node = reshape_qkv
            num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q)

            if num_heads <= 0 or hidden_size <= 0 or (hidden_size % num_heads) != 0:
                logger.debug("fuse_attention: failed to detect num_heads or hidden_size")
                return

            attention_parent = self.model.get_parent(add_qk, 1, None)
            reshape_parent_0 = self.model.get_parent(attention_parent, 0, None)
            transpose_parent = self.model.get_parent(reshape_parent_0, 0, None)
            pos_k = self.model.get_parent(transpose_parent, 1, None)

            new_node = None
            if decoder_attention_with_past or decoder_cross_attention or decoder_cross_attention_with_past:
                # Note: Decoder attention with past key and past value is fused as multihead attention
                # rather than attention because multihead attention supports separate past key and past
                # value whereas attention supports concatenated past key and past value.
                new_node = (
                    self.create_multihead_attention_node(
                        matmul_q,
                        matmul_k if decoder_cross_attention or decoder_attention_with_past else past_k,
                        matmul_v if decoder_cross_attention or decoder_attention_with_past else past_v,
                        add_q,
                        add_k if decoder_cross_attention or decoder_attention_with_past else None,
                        add_v if decoder_cross_attention or decoder_attention_with_past else None,
                        num_heads,
                        hidden_size,
                        attention_last_node.output[0],
                        # key_padding_mask=self.cast_attention_mask(mask_add.input[1]),
                        key_padding_mask=new_add,
                        # add_qk=pos_k.input[0],
                        past_k=past_k if decoder_attention_with_past else "",
                        past_v=past_v if decoder_attention_with_past else "",
                        pos_emb=pos_k.input[0],
                        present_k=present_k,
                        present_v=present_v,
                    )
                    # if self.use_multi_head_attention
                    # else None
                )
            else:
                # Temporarily set multihead attention flag to false
                use_multi_head_attention_ground_truth = self.use_multi_head_attention
                self.use_multi_head_attention = False
                new_node = self.create_attention_node(
                    None,
                    matmul_q,
                    matmul_k,
                    matmul_v,
                    add_q,
                    add_k,
                    add_v,
                    num_heads,
                    hidden_size,
                    root_input,
                    attention_last_node.output[0],
                    add_qk_str=mask_index if decoder_attention else None,
                    past_k=past_k,
                    past_v=past_v,
                    present_k=present_k,
                    present_v=present_v,
                )
                self.use_multi_head_attention = use_multi_head_attention_ground_truth
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)

            # When using multihead attention, keep MatMul nodes in original graph
            if decoder_attention_with_past or decoder_cross_attention or decoder_cross_attention_with_past:
                if q_nodes[-1].op_type == "MatMul":
                    q_nodes.pop()
                if k_nodes[-1].op_type == "MatMul":
                    k_nodes.pop()
                if v_nodes[-1].op_type == "MatMul":
                    v_nodes.pop()
                if self.disable_multi_head_attention_bias and (
                    decoder_cross_attention or decoder_cross_attention_with_past
                ):
                    if q_nodes[-1].op_type == "Add":
                        q_nodes.pop()
                    if k_nodes[-1].op_type == "Add":
                        k_nodes.pop()
                    if v_nodes[-1].op_type == "Add":
                        v_nodes.pop()

            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes)

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            self.prune_graph = True
            print("Done with pruning")
