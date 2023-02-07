# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Union

from fusion_attention import AttentionMask, FusionAttention
from fusion_base import Fusion
from fusion_skiplayernorm import FusionSkipLayerNormalization
from onnx import NodeProto
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel

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


# It's much easier to export it with the custom op. TODO: revisit later
class FusionRelativePositionBiasBlock(Fusion):
    def __init__(self, model: OnnxModel, max_distance: int, is_bidirectional: bool):
        super().__init__(model, "RelativePositionBias", "Add")
        self.max_distance = max_distance
        self.is_bidirectional = is_bidirectional

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # Not implemented yet
        return


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
        # TODO: hardcode for now. double check later
        self.rpb_fusion = FusionRelativePositionBiasBlock(self, 32, True)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def fuse_skip_layer_norm(self):
        self.skip_layer_norm_fusion.apply()

    def postprocess(self):
        self.rpb_fusion.apply()
        self.clean_graph()
        self.prune_graph()
