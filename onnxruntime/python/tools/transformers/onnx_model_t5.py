# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Union

from fusion_attention import AttentionMask, FusionAttention
from fusion_base import Fusion
from fusion_utils import NumpyHelper
from fusion_skiplayernorm import FusionSkipLayerNormalization
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel

import numpy as np

logger = logging.getLogger(__name__)

# TODO: Support decoder self/cross attention fusion and encoder self attention fusion
class FusionT5Attention(FusionAttention):
    """
    Fuse T5 Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
    ):
        super().__init__(model, hidden_size, num_heads, attention_mask)

    def create_attention_node(
        self,
        mask_index: str,
        matmul: NodeProto,
        add: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
        add_qk_str: str,
    ) -> Union[NodeProto, None]:
        # Not implemented yet
        return None

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Not implemented yet
        return


# Attr("max_distance", "Max distance", AttributeProto::INT)
# Attr("is_bidirectional", "Default value is 0.", AttributeProto::INT, static_cast<int64_t>(0))
# Input(0, "bias_table", "2D input tensor with shape (num_buckets, num_heads), COL-major(See UT for example)", "T")
# Input(1, "query_length", "The length of query. Self Attention requires query_length = key_length", "U")
# Input(2, "key_length", "The length of key.", "U")
# Output(0, "output", "4D output tensor with shape (1, num_heads, sequence_length, sequence_length)", "T")
class FusionRelativePositionBiasBlock(Fusion):
    def __init__(self, model: OnnxModel, max_distance: int):
        super().__init__(model, "RelativePositionBias", ["Add", "Slice"])
        self.max_distance = max_distance
        # bidirectional=(not self.is_decoder)
        self.is_bidirectional = False

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if node.op_type != "Add" and node.op_type != "Slice":
            return

        compute_bias_nodes = self.model.match_parent_path(
            node,
            ["Unsqueeze", "Transpose", "Gather", "Where"],
            [0, 0, 0, 1]
        )
        if compute_bias_nodes is None:
            return

        gather = compute_bias_nodes[2]
        where = compute_bias_nodes[-1]
        unsqueeze = compute_bias_nodes[0]

        compute_buckets_nodes = self.model.match_parent_path(
            where,
            ["Min", "ConstantOfShape", "Shape", "Add", "Cast", "Mul", "Div", "Log", "Div", "Cast", "Neg"],
            [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        if compute_buckets_nodes is None:
            return

        neg = compute_buckets_nodes[-1]

        less_nodes = self.model.match_parent_path(
            where,
            ["Less"],
            [0]
        )
        if less_nodes is None:
            return

        less = less_nodes[0]
        if (less.input[0] != neg.output[0]):
            return

        range_nodes = self.model.match_parent_path(
            neg,
            ["Min", "ConstantOfShape", "Shape", "Sub", "Unsqueeze", "Range"],
            [0, 1, 0, 0, 0, 0]
        )
        if range_nodes is None:
            return
        range = range_nodes[-1]

        self.nodes_to_remove.extend(compute_bias_nodes)
        self.nodes_to_remove.extend(compute_buckets_nodes)
        self.nodes_to_remove.extend(less_nodes)
        self.nodes_to_remove.extend(range_nodes)

        table_weight_i = self.model.get_initializer(gather.input[0])
        table_weight = NumpyHelper.to_array(table_weight_i)
        table_weight_t = np.transpose(table_weight)
        bias_table = helper.make_tensor(
            name="bias_table_weight",
            data_type=TensorProto.FLOAT,
            dims=[np.shape(table_weight)[0], np.shape(table_weight)[1]],
            vals=table_weight_t.flatten().tolist(),
        )

        self.model.add_initializer(bias_table, self.this_graph_name)
        inputs = [bias_table.name, range.input[1], range.input[1]]
        outputs = [unsqueeze.output[0]]
        rpb_node = helper.make_node(
            "RelativePositionBias",
            inputs=inputs,
            outputs=outputs,
            name=self.model.create_node_name("RelativePositionBias", name_prefix="RPB"),
        )
        rpb_node.domain = "com.microsoft"
        rpb_node.attribute.extend([helper.make_attribute("max_distance", self.max_distance)])
        rpb_node.attribute.extend([helper.make_attribute("is_bidirectional", self.is_bidirectional)])

        self.nodes_to_add.append(rpb_node)
        self.node_name_to_graph_name[rpb_node.name] = self.this_graph_name


class FusionSkipSimplifiedLayerNormalization(FusionSkipLayerNormalization):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "SkipSimplifiedLayerNormalization", "SimplifiedLayerNormalization")
        self.shape_infer_helper = self.model.infer_runtime_shape(
            {"batch_size": 2, "seq_len": 1, "encode_sequence_length": 8, "past_decode_sequence_length": 4}, update=True
        )

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        super().fuse(node, input_name_to_nodes, output_name_to_node)


class T5OnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionT5Attention(self, self.hidden_size, self.num_heads, self.attention_mask)
        self.skip_layer_norm_fusion = FusionSkipSimplifiedLayerNormalization(self)
        self.rpb_fusion = FusionRelativePositionBiasBlock(self, 128)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def fuse_skip_layer_norm(self):
        self.skip_layer_norm_fusion.apply()

    def postprocess(self):
        self.rpb_fusion.apply()
        self.clean_graph()
        self.prune_graph()
