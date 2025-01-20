# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from fusion_attention import AttentionMask, FusionAttention
from onnx_model import OnnxModel

logger = logging.getLogger(__name__)


class FusionConformerAttention(FusionAttention):
    """
    Fuse Conformer Attention subgraph into one MultiHeadAttention node.
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
        qkv_nodes_1 = self.model.match_parent_path(
            normalize_node,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [1, 1, 0, 0, 0],
        )
        qkv_nodes_2 = self.model.match_parent_path(
            normalize_node,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [1, 0, 0, 0, 0],
        )

        qkv_nodes = None
        if qkv_nodes_1 is not None:
            (_, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes_1
            qkv_nodes = qkv_nodes_1
        elif qkv_nodes_2 is not None:
            (_, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes_2
            qkv_nodes = qkv_nodes_2
        else:
            logger.debug("fuse_conformer_attention: failed to match qkv path")
            return

        v_nodes_1 = self.model.match_parent_path(
            matmul_qkv,
            ["Concat", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 1, 0, 0, 1],
        )
        v_nodes_2 = self.model.match_parent_path(
            matmul_qkv,
            ["Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 0, 0],
        )

        v_nodes = None
        add_v = None
        past_v, present_v = "", ""
        if v_nodes_1 is not None:
            (concat_v, _, _, add_v, matmul_v) = v_nodes_1
            concat_parent = self.model.get_parent(concat_v, 0, None)
            present_v = concat_v.output[0]
            past_v = concat_parent.output[0]
            v_nodes = v_nodes_1
        elif v_nodes_2 is not None:
            (_, _, add_v, matmul_v) = v_nodes_2
            v_nodes = v_nodes_2
        else:
            logger.debug("fuse_conformer_attention: failed to match v path")
            return

        qk_nodes_1 = self.model.match_parent_path(
            matmul_qkv,
            ["Softmax", "Add", "MatMul"],
            [0, 0, 0],
        )
        qk_nodes_2 = self.model.match_parent_path(
            matmul_qkv,
            ["Where", "Softmax", "Where", "Add", "MatMul"],
            [0, 2, 0, 2, 0],
        )

        qk_nodes = None
        add_qk = None
        attn_mask = ""
        if qk_nodes_1 is not None:
            _, add_qk, matmul_qk = qk_nodes_1
            qk_nodes = qk_nodes_1
        elif qk_nodes_2 is not None:
            _, _, where_qk, add_qk, matmul_qk = qk_nodes_2
            mask_nodes = self.model.match_parent_path(
                where_qk,
                ["Equal", "Cast", "Unsqueeze", "Expand"],
                [0, 0, 0, 0],
            )
            if mask_nodes is not None:
                attn_mask = mask_nodes[-1].output[0]
            qk_nodes = qk_nodes_2
        else:
            logger.debug("fuse_conformer_attention: failed to match qk path")
            return

        q_nodes_1 = self.model.match_parent_path(
            matmul_qk,
            ["Div", "Transpose", "Reshape", "Add", "MatMul"],
            [0, 0, 0, 0, 1],
        )
        q_nodes_2 = self.model.match_parent_path(
            matmul_qk,
            ["Mul", "Transpose", "Reshape", "Add", "MatMul"],
            [0, 0, 0, 0, 0],
        )

        q_nodes = None
        if q_nodes_1 is not None:
            _, _, reshape_q, add_q, matmul_q = q_nodes_1
            q_nodes = q_nodes_1
        elif q_nodes_2 is not None:
            _, _, reshape_q, add_q, matmul_q = q_nodes_2
            q_nodes = q_nodes_2
        else:
            logger.debug("fuse_conformer_attention: failed to match q path")
            return

        k_nodes_1 = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Concat", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 1, 0, 0, 1],
        )
        k_nodes_2 = self.model.match_parent_path(
            matmul_qk,
            ["Transpose", "Transpose", "Reshape", "Add", "MatMul"],
            [1, 0, 0, 0, 0],
        )

        k_nodes = None
        matmul_k = None
        past_k, present_k = "", ""
        if k_nodes_1 is not None:
            _, concat_k, _, _, add_k, matmul_k = k_nodes_1
            concat_parent = self.model.get_parent(concat_k, 0, None)
            past_k = concat_parent.output[0]
            present_k = concat_k.output[0]
            k_nodes = k_nodes_1
        elif k_nodes_2 is not None:
            _, _, _, add_k, matmul_k = k_nodes_2
            k_nodes = k_nodes_2
        else:
            logger.debug("fuse_conformer_attention: failed to match k path")
            return

        num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q)
        if num_heads <= 0 or hidden_size <= 0 or (hidden_size % num_heads) != 0:
            logger.debug("fuse_conformer_attention: failed to detect num_heads or hidden_size")
            return

        new_node = None
        use_packed_attention_op = matmul_q.input[0] == matmul_k.input[0] and matmul_k.input[0] == matmul_v.input[0]
        if use_packed_attention_op:
            # Self-attention, use Attention op
            new_node = self.create_attention_node(
                mask_index=attn_mask,
                q_matmul=matmul_q,
                k_matmul=matmul_k,
                v_matmul=matmul_v,
                q_add=add_q,
                k_add=add_k,
                v_add=add_v,
                num_heads=num_heads,
                hidden_size=hidden_size,
                first_input=matmul_q.input[0],
                output=reshape_qkv.output[0],
                add_qk_str=add_qk.input[1],
                past_k=past_k,
                past_v=past_v,
                present_k=present_k,
                present_v=present_v,
            )
        else:
            new_node = self.create_multihead_attention_node(
                q_matmul=matmul_q,
                k_matmul=matmul_k,
                v_matmul=matmul_v,
                q_add=add_q,
                k_add=add_k,
                v_add=add_v,
                num_heads=num_heads,
                hidden_size=hidden_size,
                output=reshape_qkv.output[0],
                key_padding_mask=attn_mask,
                add_qk=add_qk.input[1],
                past_k=past_k,
                past_v=past_v,
                present_k=present_k,
                present_v=present_v,
            )

        if new_node is None:
            logger.debug("fuse_conformer_attention: MultiHeadAttention node creation failed")
            return

        self.increase_counter(new_node.op_type)
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([reshape_qkv, transpose_qkv, matmul_qkv])
        self.nodes_to_remove.extend(qk_nodes)

        # When using MultiHeadAttention, keep MatMul nodes unfused in original graph
        if not use_packed_attention_op:
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
