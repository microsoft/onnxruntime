# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from fusion_attention import AttentionMask, FusionAttention
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

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [1, 1, 0, 0, 0],
        )
        if qkv_nodes is not None:
            (
                _,
                _,
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

        # graph_input_names = set([node.name for node in self.model.graph().input])
        # graph_output_names = set([node.name for node in self.model.graph().output])

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Concat", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 1, 0, 0, 1],
        )

        add_v = None
        if v_nodes is not None:
            (concat_v, _, _, add_v, matmul_v) = v_nodes
            concat_parent = self.model.get_parent(concat_v, 0, None)
            present_v = concat_v.output[0]
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
            _, _, reshape_q, add_q, matmul_q = q_nodes
        else:
            return

        k_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Concat", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 1, 0, 0, 1],
        )

        matmul_k = None
        if k_nodes is not None:
            _, concat_k, _, _, add_k, matmul_k = k_nodes
            concat_parent = self.model.get_parent(concat_k, 0, None)
            past_k = concat_parent.output[0]
            present_k = concat_k.output[0]
        else:
            return

        attention_last_node = reshape_qkv
        num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q)

        if num_heads <= 0 or hidden_size <= 0 or (hidden_size % num_heads) != 0:
            logger.debug("fuse_attention: failed to detect num_heads or hidden_size")
            return

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
            add_qk=add_qk.input[1],
            past_k=past_k,
            past_v=past_v,
            present_k=present_k,
            present_v=present_v,
            kv_cache_name_match=False,
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

        self.nodes_to_remove.extend(k_nodes)
        self.nodes_to_remove.extend(v_nodes)

        # Use prune graph to remove mask nodes since they are shared by all attention nodes.
        self.prune_graph = True
