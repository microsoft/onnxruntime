# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from typing import Optional

from fusion_attention_unet import FusionAttentionUnet
from fusion_bias_add import FusionBiasAdd
from fusion_options import FusionOptions
from import_utils import is_installed
from onnx import ModelProto
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)


class MmditOnnxModel(BertOnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """Initialize Multimodal Diffusion Transformer (MMDiT) ONNX Model.

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attention heads. Defaults to 0 (detect the parameter automatically).
            hidden_size (int, optional): hidden dimension. Defaults to 0 (detect the parameter automatically).
        """
        assert (num_heads == 0 and hidden_size == 0) or (num_heads > 0 and hidden_size % num_heads == 0)

        super().__init__(model, num_heads=num_heads, hidden_size=hidden_size)

    def preprocess(self):
        self.remove_useless_div()

    def postprocess(self):
        self.prune_graph()
        self.remove_unused_constant()

    def remove_useless_div(self):
        """Remove Div by 1"""
        div_nodes = [node for node in self.nodes() if node.op_type == "Div"]

        nodes_to_remove = []
        for div in div_nodes:
            if self.find_constant_input(div, 1.0) == 1:
                nodes_to_remove.append(div)

        for node in nodes_to_remove:
            self.replace_input_of_all_nodes(node.output[0], node.input[0])

        if nodes_to_remove:
            self.remove_nodes(nodes_to_remove)
            logger.info("Removed %d Div nodes", len(nodes_to_remove))

    def fuse_multi_head_attention(self, options: Optional[FusionOptions] = None):
        # Self Attention
        self_attention_fusion = FusionAttentionUnet(
            self,
            self.hidden_size,
            self.num_heads,
            is_cross_attention=False,
            enable_packed_qkv=False,
            enable_packed_kv=False,
        )
        self_attention_fusion.apply()

        # Cross Attention
        cross_attention_fusion = FusionAttentionUnet(
            self,
            self.hidden_size,
            self.num_heads,
            is_cross_attention=True,
            enable_packed_qkv=False,
            enable_packed_kv=False,
        )
        cross_attention_fusion.apply()

    def fuse_bias_add(self):
        fusion = FusionBiasAdd(self)
        fusion.apply()

    def optimize(self, options: Optional[FusionOptions] = None):
        if is_installed("tqdm"):
            import tqdm
            from tqdm.contrib.logging import logging_redirect_tqdm

            with logging_redirect_tqdm():
                steps = 18
                progress_bar = tqdm.tqdm(range(steps), initial=0, desc="fusion")
                self._optimize(options, progress_bar)
        else:
            logger.info("tqdm is not installed. Run optimization without progress bar")
            self._optimize(options, None)

    def _optimize(self, options: Optional[FusionOptions] = None, progress_bar=None):
        if (options is not None) and not options.enable_shape_inference:
            self.disable_shape_inference()

        self.utils.remove_identity_nodes()
        if progress_bar:
            progress_bar.update(1)

        # Remove cast nodes that having same data type of input and output based on symbolic shape inference.
        self.utils.remove_useless_cast_nodes()
        if progress_bar:
            progress_bar.update(1)

        if (options is None) or options.enable_layer_norm:
            self.fuse_layer_norm()
        if progress_bar:
            progress_bar.update(1)

        self.preprocess()
        if progress_bar:
            progress_bar.update(1)

        self.fuse_reshape()
        if progress_bar:
            progress_bar.update(1)


        if (options is None) or options.enable_attention:
            self.fuse_multi_head_attention(options)
        if progress_bar:
            progress_bar.update(1)

        if (options is None) or options.enable_skip_layer_norm:
            self.fuse_skip_layer_norm()
        if progress_bar:
            progress_bar.update(1)

        self.fuse_shape()
        if progress_bar:
            progress_bar.update(1)

        # Remove reshape nodes that having same shape of input and output based on symbolic shape inference.
        self.utils.remove_useless_reshape_nodes()
        if progress_bar:
            progress_bar.update(1)

        if (options is None) or options.enable_bias_skip_layer_norm:
            # Fuse SkipLayerNormalization and Add Bias before it.
            self.fuse_add_bias_skip_layer_norm()
        if progress_bar:
            progress_bar.update(1)

        self.postprocess()
        if progress_bar:
            progress_bar.update(1)

        logger.info(f"opset version: {self.get_opset_version()}")

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        ops = [
            "MultiHeadAttention",
            "LayerNormalization",
            "SkipLayerNormalization",
        ]

        for op in ops:
            nodes = self.get_nodes_by_op_type(op)
            op_count[op] = len(nodes)

        logger.info(f"Optimized operators:{op_count}")
        return op_count
