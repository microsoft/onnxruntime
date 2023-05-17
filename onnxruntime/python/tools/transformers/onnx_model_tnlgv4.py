# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Union

from fusion_attention import AttentionMask
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel
from fusion_base import Fusion
from fusion_utils import FusionUtils
import numpy as np
logger = logging.getLogger(__name__)


class FusionTNLGV4Attention(Fusion):
    """
    Fuse GPT-2 Attention with past state subgraph into one Attention node.
    """

    def __init__(self, model: OnnxModel, num_heads: int):
        super().__init__(model, "Attention", ["MatMul"])
        self.num_heads = num_heads

    def create_attention_node(
        self,
        fc_weight,
        fc_bias,
        input,
        output,
        mask,
        past,
        present,
    ):
        attention_node_name = self.model.create_node_name("TNLGV4Attention")
        attention_node = helper.make_node(
            "Attention",
            inputs=[input, fc_weight, fc_bias, mask, past],
            outputs=[output, present],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend(
            [
                helper.make_attribute("num_heads", self.num_heads),
                helper.make_attribute("unidirectional", 1),
                helper.make_attribute("do_rotary", 1),
                helper.make_attribute("scale", 0.0078125),
            ]
        )

        # attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        self.nodes_to_add.extend([attention_node])
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # note: this is not an exact match. experiment purpose only.
        stem_nodes = self.model.match_parent_path(
            node,
            ['Reshape', 'Transpose', 'Reshape', 'MatMul', 'Reshape', 'Softmax', 'Where', 'Reshape', 'Add', 'Mul', 'MatMul']
        )
        if stem_nodes is None:
            return

        qk_matmul_node = stem_nodes[-1]

        qkv_matmul_nodes = self.model.match_parent_path(
            qk_matmul_node,
            ['Transpose', 'Reshape', 'Add', 'Mul', 'Split', 'Reshape', 'Add', 'MatMul'],
            [0, 0, 0, 0, 0, 0, 0, 1]
        )
        if qkv_matmul_nodes is None:
            return

        where_node = stem_nodes[-5]
        attn_mask_nodes = self.model.match_parent_path(
            where_node,
            ['Slice', 'Slice']
        )
        if attn_mask_nodes is None:
            return

        past_nodes = self.model.match_parent_path(
            qk_matmul_node,
            ['Transpose', 'Reshape', 'Concat', 'Squeeze', 'Split'],
            [1, 0, 0, 0, 0],
        )
        if past_nodes is None:
            return

        concat_node = past_nodes[-3]
        unsqueeze_node = self.model.find_first_child_by_type(concat_node, 'Unsqueeze')
        if unsqueeze_node is None:
            return
        concat_node_2 = self.model.find_first_child_by_type(unsqueeze_node, 'Concat')
        if concat_node_2 is None:
            return

        input = qkv_matmul_nodes[-1].input[0]
        attn_mask = attn_mask_nodes[-1].input[0]
        fc_weights = qkv_matmul_nodes[-1].input[1]
        fc_bias = qkv_matmul_nodes[-2].input[0]
        past = past_nodes[-1].input[0]
        output = node.input[0]
        present = concat_node_2.output[0]

        self.create_attention_node(fc_weights, fc_bias, input, output, attn_mask, past, present)

        self.nodes_to_remove.extend(stem_nodes)
        self.nodes_to_remove.extend(qkv_matmul_nodes)
        self.nodes_to_remove.extend(attn_mask_nodes)
        self.nodes_to_remove.extend(past_nodes)
        self.nodes_to_remove.extend([concat_node_2])

        # todo_1: append transpose before and after the attention node
        self.prune_graph = True


class Tnlgv4OnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionTNLGV4Attention(self, self.num_heads)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def preprocess(self):
        self.utils.remove_useless_cast_nodes_in_fp16_model()
