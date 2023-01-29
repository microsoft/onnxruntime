# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Union

from fusion_attention import AttentionMask, FusionAttention
from fusion_utils import NumpyHelper
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel
from fusion_base import Fusion
import numpy as np

logger = logging.getLogger(__name__)

#python optimizer.py --input /home/wy/Turing/tulr/model.onnx --output /home/wy/Turing/tulr/opt/model.onnx --model_type tulr --num_heads 16 --hidden_size 1024 --use_external_data_format

class FusionTulrAttention(FusionAttention):
    """
    Turing
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
        return

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        return

# Attr("max_distance", "Max distance", AttributeProto::INT)
# Attr("is_bidirectional", "Default value is 0.", AttributeProto::INT, static_cast<int64_t>(0))
# Input(0, "bias_table", "2D input tensor with shape (num_buckets, num_heads), COL-major(See UT for example)", "T")
# Input(1, "query_length", "The length of query. Self Attention requires query_length = key_length", "U")
# Input(2, "key_length", "The length of key.", "U")
# Output(0, "output", "4D output tensor with shape (1, num_heads, sequence_length, sequence_length)", "T")
class FusionRelativePositionBiasBlock(Fusion):
    def __init__(self, model: OnnxModel, max_distance: int, is_bidirectional: bool):
        super().__init__(model, "RelativePositionBias", "GatherElements")
        self.max_distance = 128
        self.is_bidirectional = 1

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        stem_nodes = self.model.match_parent_path(
            node,
            ["Expand", "Where", "Equal", "Concat", "Unsqueeze", "Gather", "Shape", "Sub", "Unsqueeze", "Expand", "Unsqueeze", "Range"],
        )
        if stem_nodes is None:
            return

        expand = stem_nodes[0]
        range = stem_nodes[-1]
        rpb_nodes = self.model.match_parent_path(
            expand,
            ["Unsqueeze", "Unsqueeze", "Gemm"]
        )
        if rpb_nodes is None:
            return

        gemm = rpb_nodes[-1]

        self.nodes_to_remove.extend(stem_nodes)
        self.nodes_to_remove.extend(rpb_nodes)

        table_weight = self.model.get_initializer(gemm.input[0])
        table_weight_np = NumpyHelper.to_array(table_weight)
        bias_table = helper.make_tensor(
            name="bias_table_weight",
            data_type=TensorProto.FLOAT,
            dims=[np.shape(table_weight_np)[1], np.shape(table_weight_np)[0]],
            vals=table_weight_np.flatten().tolist(),
        )
        self.model.add_initializer(bias_table, self.this_graph_name)
        inputs = [bias_table.name, range.input[1], range.input[1]]
        outputs = [node.output[0]]
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



class TulrOnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionTulrAttention(self, self.hidden_size, self.num_heads, self.attention_mask)
        self.rpb_fusion = FusionRelativePositionBiasBlock(self, 32, True)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def postprocess(self):
        self.rpb_fusion.apply()
        self.clean_graph()
        self.prune_graph()
