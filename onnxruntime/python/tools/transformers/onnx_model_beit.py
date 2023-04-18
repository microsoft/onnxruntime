# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import Optional

from fusion_attention_beit import FusionAttentionBeit
from fusion_attention_beit_midas import FusionAttentionBeitMidas
from fusion_options import FusionOptions
from fusion_utils import FusionUtils
from onnx import ModelProto
from onnx_model_bert import BertOnnxModel

logger = getLogger(__name__)


class BeitOnnxModel(BertOnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """Initialize BEit ONNX Model.

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attention heads. Defaults to 0 (detect the parameter automatically).
            hidden_size (int, optional): hidden dimension. Defaults to 0 (detect the parameter automatically).
        """
        assert (num_heads == 0 and hidden_size == 0) or (num_heads > 0 and hidden_size % num_heads == 0)

        super().__init__(model)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attention_fusion = FusionAttentionBeit(self, self.hidden_size, self.num_heads)
        self.utils = FusionUtils(self)

    def fuse_attention(self):
        self.attention_fusion.apply()

        # When attention pattern of huggingface beit model is not matched, try match MiDas style
        if not self.attention_fusion.nodes_to_add:
            self.attention_fusion = FusionAttentionBeitMidas(self, self.hidden_size, self.num_heads)
            self.attention_fusion.apply()

    def optimize(self, options: Optional[FusionOptions] = None, add_dynamic_axes: bool = False):
        if (options is not None) and not options.enable_shape_inference:
            self.disable_shape_inference()

        self.utils.remove_identity_nodes()

        # Remove cast nodes that having same data type of input and output based on symbolic shape inference.
        self.utils.remove_useless_cast_nodes()

        if (options is None) or options.enable_layer_norm:
            self.fuse_layer_norm()

        if (options is None) or options.enable_gelu:
            self.fuse_gelu()

        self.preprocess()

        self.fuse_reshape()

        if (options is None) or options.enable_skip_layer_norm:
            self.fuse_skip_layer_norm()

        if (options is None) or options.enable_attention:
            self.fuse_attention()

        self.fuse_shape()

        # Remove reshape nodes that having same shape of input and output based on symbolic shape inference.
        self.utils.remove_useless_reshape_nodes()

        self.postprocess()

        # Bias fusion is done after postprocess to avoid extra Reshape between bias and Gelu/FastGelu/SkipLayerNormalization
        if (options is None) or options.enable_bias_gelu:
            # Fuse Gelu and Add Bias before it.
            self.fuse_bias_gelu(is_fastgelu=True)
            self.fuse_bias_gelu(is_fastgelu=False)

        if (options is None) or options.enable_bias_skip_layer_norm:
            # Fuse SkipLayerNormalization and Add Bias before it.
            self.fuse_add_bias_skip_layer_norm()

        if options is not None and options.enable_gelu_approximation:
            self.gelu_approximation()

        self.remove_unused_constant()

        # Use symbolic batch dimension in input and output.
        if add_dynamic_axes:
            self.use_dynamic_axes()

        logger.info(f"opset version: {self.get_opset_version()}")

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        ops = [
            "MultiHeadAttention",
            "Gelu",
            "FastGelu",
            "BiasGelu",
            "LayerNormalization",
            "SkipLayerNormalization",
        ]
        for op in ops:
            nodes = self.get_nodes_by_op_type(op)
            op_count[op] = len(nodes)

        logger.info(f"Optimized operators:{op_count}")
        return op_count

    def is_fully_optimized(self):
        """
        Returns True when the model is fully optimized.
        """
        op_count = self.get_fused_operator_statistics()
        attention = op_count["MultiHeadAttention"]
        gelu = op_count["Gelu"] + op_count["BiasGelu"] + op_count["FastGelu"]
        layer_norm = op_count["LayerNormalization"] + op_count["SkipLayerNormalization"]
        is_perfect = (attention > 0) and (attention == gelu) and (layer_norm >= 2 * attention)

        if layer_norm == 0:
            logger.debug("Layer Normalization not fused")

        if gelu == 0:
            logger.debug("Gelu/FastGelu not fused")

        if attention == 0:
            logger.warning("Attention not fused")

        return is_perfect
