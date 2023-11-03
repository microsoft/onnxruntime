# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

import numpy as np
from fusion_attention import AttentionMask, FusionAttention
from onnx import TensorProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = logging.getLogger(__name__)


class FusionTransducerAttention(FusionAttention):
    """
    Fuse Transducer Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
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

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [1, 1, 0, 0, 0],
        )
        if qkv_nodes is not None:
            (
                add_out,
                matmul_out,
                reshape_qkv,
                transpose_qkv,
                matmul_qkv,
            ) = qkv_nodes
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
            [1, 1, 0, 0, 1],
        )

        past_v, present_v = "", ""
        reshape_v_2, add_v = None, None
        if v_nodes is not None:
            (concat_v, transpose_v, reshape_v, add_v, matmul_v) = v_nodes
            concat_children = self.model.get_children(concat_v, None)
            gather_children = self.model.get_children(concat_children[1], None)
            concat_parent = self.model.get_parent(concat_v, 0, None)
            gather_parent = self.model.get_parent(concat_parent, 0, None)
            # present_v = gather_children[0].input[0]
            present_v = concat_v.output[0]
            print("Concat v = ", concat_v)
            past_v = concat_parent.output[0]
        else:
            logger.debug("fuse_attention: failed to match v path")
            return

        qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "Add", "MatMul"], [0, 0, 0])

        if qk_nodes is not None:
            _, add_qk, matmul_qk = qk_nodes
        else:
            return

        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Div", "Transpose", "Reshape", "Add", "MatMul"],
            [0, 0, 0, 0, 1],
        )
        if q_nodes is not None:
            div_q, transpose_q, reshape_q, add_q, matmul_q = q_nodes
        else:
            return

        k_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Concat", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 1, 0, 0, 1],
        )

        past_k, present_k = "", ""
        reshape_k_2, reshape_k_1, matmul_k = None, None, None
        if k_nodes is not None:
            _, concat_k, _, reshape_k, add_k, matmul_k = k_nodes
            concat_parent = self.model.get_parent(concat_k, 0, None)
            transpose_parent = self.model.get_parent(concat_parent, 0, None)
            concat_children = self.model.get_children(concat_k, None)
            transpose_children = self.model.get_children(concat_children[1], None)
            print("Transpose children = ", concat_k)
            gather_children = self.model.get_children(transpose_children[0], None)
            past_k = concat_parent.output[0]
            # present_k = concat_children[1].output[0]
            present_k = concat_k.output[0]
        else:
            return

        print("After this")

        # if k_nodes:
        #     # Create empty Add node for attention graph
        #     bias_dim = self.model.get_initializer(add_v.input[0]).dims[0]
        #     empty_bias_name = "empty_bias"
        #     empty_tensor = self.model.get_initializer(empty_bias_name)
        #     if empty_tensor is None:
        #         self.add_initializer(
        #             empty_bias_name,
        #             TensorProto.FLOAT,
        #             dims=[bias_dim],
        #             vals=np.array([0.0] * bias_dim, dtype=np.float32),
        #         )

        #     add_name = self.model.create_node_name("Add")
        #     add_k = helper.make_node("Add", [empty_bias_name, matmul_k.output[0]], [reshape_k.name], add_name)

        attention_last_node = reshape_qkv
        num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q)

        if num_heads <= 0 or hidden_size <= 0 or (hidden_size % num_heads) != 0:
            print("fuse_attention: failed to detect num_heads or hidden_size")
            logger.debug("fuse_attention: failed to detect num_heads or hidden_size")
            return

        # attention_parent = self.model.get_parent(add_qk, 1, None)
        # reshape_parent_0 = self.model.get_parent(attention_parent, 0, None)
        # transpose_parent = self.model.get_parent(reshape_parent_0, 0, None)
        # pos_k = self.model.get_parent(transpose_parent, 1, None)

        attention_parent = self.model.get_parent(add_qk, 1, None)
        reshape_parent_0 = self.model.get_parent(attention_parent, 0, None)
        # reshape_parent_1 = self.model.get_parent(attention_parent, 1, None)
        transpose_parent = self.model.get_parent(reshape_parent_0, 0, None)
        pos_k = self.model.get_parent(transpose_parent, 1, None)
        gather_pos_k = self.model.get_parent(pos_k, 0, None)
        add_pos_k = self.model.get_parent(gather_pos_k, 1, None)
        where_0 = self.model.get_parent(add_pos_k, 0, None)
        where_1 = self.model.get_parent(where_0, 2, None)
        sub_pos_k = self.model.get_parent(where_1, 2, None)
        usq_pos_k = self.model.get_parent(sub_pos_k, 0, None)
        range_pos_k = self.model.get_parent(usq_pos_k, 0, None)
        gather_0 = self.model.get_parent(range_pos_k, 0, None)
        shape_pos = self.model.get_parent(gather_0, 0, None)

        k_index, v_index = "index_0", "index_1"
        k_dim = self.model.get_initializer(k_index)
        k_dim = numpy_helper.from_array(np.array(1, dtype="int64"), name=k_index)
        self.model.add_initializer(k_dim, self.this_graph_name)

        # Create nodes to index kv_node
        gather_k_name = self.model.create_node_name("Gather")
        gather_pos_k_name = "Gather_Batch_size"
        # gather_v_name = self.model.create_node_name("Gather")
        present_pos_k = helper.make_node(
            "Gather",
            inputs=[gather_0.input[0], k_index],
            outputs=[gather_pos_k_name],
            name=gather_k_name,
            axis=0,
        )

        self.nodes_to_add.append(present_pos_k)
        self.node_name_to_graph_name[gather_k_name] = self.this_graph_name

        constant_unsq_1 = helper.make_tensor(
            name="Constant_0",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0],
        )
        self.model.add_initializer(constant_unsq_1, self.this_graph_name)

        unsqueeze_0 = self.model.create_node_name("Unsqueeze")
        unsqueeze_name_0 = "Unsqueeze_posK_0"
        unsqueeze_posK_0 = helper.make_node(
            "Unsqueeze",
            inputs=[gather_pos_k_name, constant_unsq_1.name],
            outputs=[unsqueeze_name_0],
            name=unsqueeze_0,
        )
        self.nodes_to_add.append(unsqueeze_posK_0)
        self.node_name_to_graph_name[unsqueeze_0] = self.this_graph_name

        constant_unsq_2 = helper.make_tensor(
            name="Constant_1",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0],
        )
        self.model.add_initializer(constant_unsq_2, self.this_graph_name)

        unsqueeze_1 = self.model.create_node_name("Unsqueeze")
        unsqueeze_name_1 = "Unsqueeze_posK_1"
        unsqueeze_posK_1 = helper.make_node(
            "Unsqueeze",
            inputs=[pos_k.input[0], constant_unsq_2.name],
            outputs=[unsqueeze_name_1],
            name=unsqueeze_1,
        )
        self.nodes_to_add.append(unsqueeze_posK_1)
        self.node_name_to_graph_name[unsqueeze_1] = self.this_graph_name

        constant_concat = helper.make_tensor(
            name="Constant_Concat",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1],
        )
        self.model.add_initializer(constant_concat, self.this_graph_name)

        constant_concat_1 = helper.make_tensor(
            name="Constant_Concat_1",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1],
        )
        self.model.add_initializer(constant_concat_1, self.this_graph_name)

        constant_concat_2 = helper.make_tensor(
            name="Constant_Concat_2",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1],
        )
        self.model.add_initializer(constant_concat_2, self.this_graph_name)

        concat_name_new = self.model.create_node_name("Concat")
        concat_n = "Concat_Expand"
        concat_expand = helper.make_node(
            "Concat",
            inputs=[unsqueeze_name_0, constant_concat.name, constant_concat_1.name, constant_concat_2.name],
            outputs=[concat_n],
            name=concat_name_new,
            axis=0,
        )
        self.nodes_to_add.append(concat_expand)
        self.node_name_to_graph_name[concat_name_new] = self.this_graph_name

        constant_expand_t = helper.make_tensor(
            name="Constant_Expand_t",
            data_type=TensorProto.INT64,
            dims=[2],
            vals=[1, 1],
        )
        self.model.add_initializer(constant_expand_t, self.this_graph_name)

        expand_pos_k = self.model.create_node_name("Expand")
        expand_pos_k_name = "Positional_embed_expand"
        posk_expand = helper.make_node(
            "Expand",
            inputs=[unsqueeze_name_1, concat_n],
            outputs=[expand_pos_k_name],
            name=expand_pos_k,
        )
        self.nodes_to_add.append(posk_expand)
        self.node_name_to_graph_name[expand_pos_k] = self.this_graph_name

        print("Created new node")

        new_node = None
        new_node = self.create_multihead_attention_node(
            matmul_q,
            matmul_k,
            matmul_v,
            add_q,
            add_k,
            add_v,
            num_heads,
            hidden_size,
            attention_last_node.output[0],
            # add_qk=add_qk.input[1],
            past_k=past_k,
            past_v=past_v,
            # positional_embedding=pos_k.input[0],
            positional_embedding=expand_pos_k_name,
            present_k=present_k,
            present_v=present_v,
        )

        if new_node is None:
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
        self.nodes_to_remove.extend(qk_nodes)

        # When using multihead attention, keep MatMul nodes in original graph
        if q_nodes[-1].op_type == "MatMul":
            q_nodes.pop()
        if k_nodes[-1].op_type == "MatMul":
            k_nodes.pop()
        if v_nodes[-1].op_type == "MatMul":
            v_nodes.pop()

        self.nodes_to_remove.extend(q_nodes)
        self.nodes_to_remove.extend(k_nodes)
        self.nodes_to_remove.extend(v_nodes)

        # Use prune graph to remove mask nodes since they are shared by all attention nodes.
        self.prune_graph = True
