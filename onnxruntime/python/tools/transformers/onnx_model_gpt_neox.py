# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Optional

import numpy as np
from fusion_attention import AttentionMask
from fusion_base import Fusion
from fusion_options import AttentionMaskFormat, FusionOptions
from onnx import TensorProto, helper
from onnx.onnx_pb import NodeProto
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)


class FusionGptNeoxAttentionNoPast(Fusion):
    """
    Fuse GPT-NeoX Attention without past state into one Attention node.
    """

    def __init__(self, model: OnnxModel, attention_mask: AttentionMask):
        super().__init__(model, "MultiHeadAttention", "Softmax", "without past")
        self.mask_filter_value = None
        self.attention_mask_tracker = attention_mask

    def merge_past_kv(self, past_k, past_v):
        if self.model.get_opset_version() < 13:
            past_k_unsqueeze_node = helper.make_node(
                "Unsqueeze",
                inputs=[past_k],
                outputs=[past_k + "_unsqueeze"],
                axes=[0],
                name=self.model.create_node_name("Unsqueeze"),
            )

            past_v_unsqueeze_node = helper.make_node(
                "Unsqueeze",
                inputs=[past_v],
                outputs=[past_v + "_unsqueeze"],
                axes=[0],
                name=self.model.create_node_name("Unsqueeze"),
            )
        else:
            # axes is moved from attribute to input
            axes_name = "ort_const_0_axes"
            if self.model.get_initializer(axes_name) is None:
                self.model.add_initializer(
                    helper.make_tensor(
                        name=axes_name,
                        data_type=TensorProto.INT64,
                        dims=[1],
                        vals=[0],
                    )
                )
            past_k_unsqueeze_node = helper.make_node(
                "Unsqueeze",
                inputs=[past_k, axes_name],
                outputs=[past_k + "_unsqueeze"],
                name=self.model.create_node_name("Unsqueeze"),
            )

            past_v_unsqueeze_node = helper.make_node(
                "Unsqueeze",
                inputs=[past_v, axes_name],
                outputs=[past_v + "_unsqueeze"],
                name=self.model.create_node_name("Unsqueeze"),
            )

        past_concat_node = helper.make_node(
            "Concat",
            inputs=[past_k + "_unsqueeze", past_v + "_unsqueeze"],
            outputs=[past_v + "_concat_" + past_v],
            axis=0,
        )

        return [past_k_unsqueeze_node, past_v_unsqueeze_node, past_concat_node]

    def split_present_kv(self, present, present_k, present_v):
        split_node = helper.make_node(
            "Split", inputs=[present], outputs=[present + "_split_0", present + "_split_1"], axis=0, split=[1, 1]
        )

        if self.model.get_opset_version() < 13:
            past_k_squeeze_node = helper.make_node(
                "Squeeze",
                inputs=[present + "_split_0"],
                outputs=[present_k],
                axes=[0],
                name=self.model.create_node_name("Squeeze"),
            )

            past_v_squeeze_node = helper.make_node(
                "Squeeze",
                inputs=[present + "_split_1"],
                outputs=[present_v],
                axes=[0],
                name=self.model.create_node_name("Squeeze"),
            )
        else:
            # axes is moved from attribute to input
            axes_name = "ort_const_0_axes"
            if self.model.get_initializer(axes_name) is None:
                self.model.add_initializer(
                    helper.make_tensor(
                        name=axes_name,
                        data_type=TensorProto.INT64,
                        dims=[1],
                        vals=[0],
                    )
                )
            past_k_squeeze_node = helper.make_node(
                "Squeeze",
                inputs=[present + "_split_0", axes_name],
                outputs=[present_k],
                name=self.model.create_node_name("Squeeze"),
            )

            past_v_squeeze_node = helper.make_node(
                "Squeeze",
                inputs=[present + "_split_1", axes_name],
                outputs=[present_v],
                name=self.model.create_node_name("Squeeze"),
            )

        return [split_node, past_k_squeeze_node, past_v_squeeze_node]

    def create_attention_node(
        self,
        matmul_before_mha: NodeProto,
        add_before_mha: NodeProto,
        reshape_before_mha: NodeProto,
        padding_mask: str,
        present_k: str,
        present_v: str,
        past_k: str = "",
        past_v: str = "",
        mask_filter_value: Optional[float] = None,
    ):
        i, x = self.model.get_constant_input(matmul_before_mha)
        if i is None:
            return

        j, b = self.model.get_constant_input(add_before_mha)
        if j is None:
            return

        # The weight is format like [NH,N*(H+H+H)], and we need to conver to format like [NH, N*H + N*H + N*H]
        _, s = self.model.get_constant_input(reshape_before_mha)
        if s is None or len(s) != 4 or s[3] % 3 != 0:
            return

        num_heads = s[2]
        head_size = int(s[3] / 3)
        if num_heads <= 0 or head_size <= 0:
            return

        y = x.reshape([x.shape[0], num_heads, 3, head_size])
        wq = y[:, :, 0, :]
        wk = y[:, :, 1, :]
        wv = y[:, :, 2, :]
        weight = np.stack((wq, wk, wv), axis=1).reshape(x.shape)

        z = b.reshape([num_heads, 3, head_size])
        bq = z[:, 0, :]
        bk = z[:, 1, :]
        bv = z[:, 2, :]
        bias = np.stack((bq, bk, bv), axis=0).reshape(b.shape)

        # TODO: use MultiHeadAttention when it supports rotary and unidirectional.
        attention_node_name = self.model.create_node_name("Attention")

        weight_tensor = helper.make_tensor(
            name=attention_node_name + "_qkv_weight",
            data_type=TensorProto.FLOAT16 if weight.dtype == np.float16 else TensorProto.FLOAT,
            dims=weight.shape,
            vals=weight.flatten().tolist(),
        )

        # Sometimes weights and bias are stored in fp16
        # if q_weight.data_type == 10:
        #     weight.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weight).astype(np.float16), weight.name))
        self.model.add_initializer(weight_tensor, self.this_graph_name)

        bias_tensor = helper.make_tensor(
            name=attention_node_name + "_qkv_bias",
            data_type=TensorProto.FLOAT16 if bias.dtype == np.float16 else TensorProto.FLOAT,
            dims=bias.shape,
            vals=bias.flatten().tolist(),
        )

        self.model.add_initializer(bias_tensor, self.this_graph_name)

        inputs = [
            matmul_before_mha.input[0],
            weight_tensor.name,
            bias_tensor.name,
            self.attention_mask_tracker.process_mask(padding_mask),
        ]

        merge_nodes = []
        if past_k:
            merge_nodes = self.merge_past_kv(past_k, past_v)
            past = merge_nodes[-1].output[0]
            inputs.append(past)

        present_name = attention_node_name + "_output_present"
        attention_node = helper.make_node(
            "Attention",
            inputs=inputs,
            outputs=[attention_node_name + "_output", present_name],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend(
            [
                helper.make_attribute("num_heads", num_heads),
                helper.make_attribute("unidirectional", 1),
                helper.make_attribute("do_rotary", 1),
            ]
        )

        split_nodes = self.split_present_kv(present_name, present_k, present_v)

        if mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(mask_filter_value))])

        nodes_to_add = [attention_node] + merge_nodes + split_nodes
        for node in nodes_to_add:
            self.node_name_to_graph_name[node.name] = self.this_graph_name

        self.nodes_to_add.extend(nodes_to_add)

        return attention_node

    def fuse(self, softmax_node, input_name_to_nodes, output_name_to_node):
        cast = self.model.find_first_child_by_type(softmax_node, "Cast", input_name_to_nodes, False)
        if cast is None:
            return

        matmul_qkv = self.model.find_first_child_by_type(cast, "MatMul", input_name_to_nodes, False)
        if matmul_qkv is None:
            return

        transpose_qkv = self.model.find_first_child_by_type(matmul_qkv, "Transpose", input_name_to_nodes, False)
        if transpose_qkv is None:
            return

        reshape_mha = self.model.find_first_child_by_type(transpose_qkv, "Reshape", input_name_to_nodes, False)
        if reshape_mha is None:
            return

        matmul_after_mha = self.model.find_first_child_by_type(reshape_mha, "MatMul", input_name_to_nodes, False)
        if matmul_after_mha is None:
            return

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Transpose", "Slice", "Reshape", "Add", "MatMul", "LayerNormalization"],
            [1, 0, 0, 0, None, 0],
        )
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (
            transpose_v,
            _slice_v,
            reshape_before_mha,
            add_before_mha,
            matmul_before_mha,
            layernorm_before_mha,
        ) = v_nodes

        present_v = transpose_v.output[0]

        mask_nodes = self.model.match_parent_path(
            softmax_node,
            ["Add", "Cast", "Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze", "Reshape"],
            [0, 1, 0, None, None, 0, 0, 0],
        )
        if mask_nodes is None:
            return

        add_mask_node = mask_nodes[0]

        _, mask_value = self.model.get_constant_input(mask_nodes[2])
        mask_filter_value = float(mask_value) if mask_value is not None else None

        padding_mask = mask_nodes[-1].input[0]

        # Here we do not match all nodes, and only some key nodes for causal mask
        causal_mask_nodes = self.model.match_parent_path(add_mask_node, ["Where", "Slice", "Slice"], [0, 0, 0])
        if causal_mask_nodes is None:
            return
        where_node = causal_mask_nodes[0]

        qk_nodes = self.model.match_parent_path(where_node, ["Reshape", "Add", "Mul", "MatMul"], [1, 0, 0, None])
        if qk_nodes is None:
            return

        matmul_qk = qk_nodes[-1]
        k_nodes = self.model.match_parent_path(
            matmul_qk,
            [
                "Transpose",
                "Reshape",
                "Concat",
                "Add",
                "Mul",
                "GatherElements",
                "Tile",
                "Expand",
                "Slice",
                "Unsqueeze",
                "Gather",
                "Shape",
                "Transpose",
                "Slice",
                "Reshape",
            ],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        )
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return

        present_k = k_nodes[2].output[0]

        if k_nodes[-1] != reshape_before_mha:
            logger.debug("fuse_attention: failed to match reshape_before_mha for k path")
            return

        q_nodes = self.model.match_parent_path(
            matmul_qk,
            ["Reshape", "Concat", "Add", "Mul", "Cast", "Slice", "Transpose", "Slice", "Reshape"],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return

        if q_nodes[-1] != reshape_before_mha:
            logger.debug("fuse_attention: failed to match reshape_before_mha for k path")
            return

        new_node = self.create_attention_node(
            matmul_before_mha,
            add_before_mha,
            reshape_before_mha,
            padding_mask,
            present_k,
            present_v,
            mask_filter_value=mask_filter_value,
        )
        if new_node is None:
            return

        matmul_after_mha.input[0] = new_node.output[0]

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([reshape_mha, transpose_qkv, matmul_qkv])

        self.prune_graph = True


class GptNeoXOnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        # We use left-side padding for generation so we cannot use 1D mask. Use 2D attention mask format.
        self.attention_mask.set_mask_format(AttentionMaskFormat.AttentionMask)

    def fuse_attention(self):
        if len(self.model.graph.input) <= 2:
            fusion = FusionGptNeoxAttentionNoPast(self, self.attention_mask)
            fusion.apply()
        # else:
        #     fusion = FusionGptNeoxAttention(self, self.num_heads)
        #     fusion.apply()
