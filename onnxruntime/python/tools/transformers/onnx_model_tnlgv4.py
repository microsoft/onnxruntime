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

    def create_attention_node(
        self,
        fc_weight,
        fc_bias,
        gemm_qkv,
        past,
        present,
        input,
        output,
        mask,
        is_unidirectional,
    ):
        attention_node_name = self.model.create_node_name("GptAttention")
        attention_node = helper.make_node(
            "Attention",
            inputs=[input, fc_weight, fc_bias, mask, past],
            outputs=[attention_node_name + "_output", present],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend(
            [
                helper.make_attribute("num_heads", self.num_heads),
                helper.make_attribute("unidirectional", 1 if is_unidirectional else 0),
            ]
        )

        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        matmul_node = helper.make_node(
            "MatMul",
            inputs=[attention_node_name + "_output", gemm_qkv.input[1]],
            outputs=[attention_node_name + "_matmul_output"],
            name=attention_node_name + "_matmul",
        )

        add_node = helper.make_node(
            "Add",
            inputs=[attention_node_name + "_matmul_output", gemm_qkv.input[2]],
            outputs=[output],
            name=attention_node_name + "_add",
        )
        self.nodes_to_add.extend([attention_node, matmul_node, add_node])
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name
        self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name
        self.node_name_to_graph_name[add_node.name] = self.this_graph_name

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        print("herehere")
        return


class Tnlgv4OnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionTNLGV4Attention(self, self.num_heads)

    def fuse_attention(self):
        self.attention_fusion.apply()
