# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser


class AttentionMaskFormat:
    MaskIndexEnd = 0
    MaskIndexEndAndStart = 1
    AttentionMask = 2
    NoMask = 3


class FusionOptions:
    """Options of fusion in graph optimization"""

    def __init__(self, model_type):
        self.enable_gelu = True
        self.enable_layer_norm = True
        self.enable_attention = True

        # Use MultiHeadAttention instead of Attention operator. The difference:
        # (1) Attention has merged weights for Q/K/V projection, which might be faster in some cases since 3 MatMul is
        #     merged into one.
        # (2) Attention could only handle self attention; MultiHeadAttention could handle both self and cross attention.
        # (3) MultiHeadAttention has only cuda implementation right now.
        self.use_multi_head_attention = False

        self.enable_skip_layer_norm = True
        self.enable_embed_layer_norm = True
        self.enable_bias_skip_layer_norm = True
        self.enable_bias_gelu = True
        self.enable_gelu_approximation = False
        self.enable_qordered_matmul = True

        self.enable_shape_inference = True
        self.enable_gemm_fast_gelu = False
        self.attention_mask_format = AttentionMaskFormat.AttentionMask

    def use_raw_attention_mask(self, use_raw_mask=True):
        if use_raw_mask:
            self.attention_mask_format = AttentionMaskFormat.AttentionMask
        else:
            self.attention_mask_format = AttentionMaskFormat.MaskIndexEnd

    def disable_attention_mask(self):
        self.attention_mask_format = AttentionMaskFormat.NoMask

    @staticmethod
    def parse(args):
        options = FusionOptions(args.model_type)
        if args.disable_gelu:
            options.enable_gelu = False
        if args.disable_layer_norm:
            options.enable_layer_norm = False
        if args.disable_attention:
            options.enable_attention = False
        if args.use_multi_head_attention:
            options.use_multi_head_attention = True
        if args.disable_skip_layer_norm:
            options.enable_skip_layer_norm = False
        if args.disable_embed_layer_norm:
            options.enable_embed_layer_norm = False
        if args.disable_bias_skip_layer_norm:
            options.enable_bias_skip_layer_norm = False
        if args.disable_bias_gelu:
            options.enable_bias_gelu = False
        if args.enable_gelu_approximation:
            options.enable_gelu_approximation = True
        if args.disable_shape_inference:
            options.enable_shape_inference = False
        if args.enable_gemm_fast_gelu:
            options.enable_gemm_fast_gelu = True
        if args.use_mask_index:
            options.use_raw_attention_mask(False)
        if args.no_attention_mask:
            options.disable_attention_mask()
        return options

    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument(
            "--disable_attention",
            required=False,
            action="store_true",
            help="disable Attention fusion",
        )
        parser.set_defaults(disable_attention=False)

        parser.add_argument(
            "--disable_skip_layer_norm",
            required=False,
            action="store_true",
            help="disable SkipLayerNormalization fusion",
        )
        parser.set_defaults(disable_skip_layer_norm=False)

        parser.add_argument(
            "--disable_embed_layer_norm",
            required=False,
            action="store_true",
            help="disable EmbedLayerNormalization fusion",
        )
        parser.set_defaults(disable_embed_layer_norm=False)

        parser.add_argument(
            "--disable_bias_skip_layer_norm",
            required=False,
            action="store_true",
            help="disable Add Bias and SkipLayerNormalization fusion",
        )
        parser.set_defaults(disable_bias_skip_layer_norm=False)

        parser.add_argument(
            "--disable_bias_gelu",
            required=False,
            action="store_true",
            help="disable Add Bias and Gelu/FastGelu fusion",
        )
        parser.set_defaults(disable_bias_gelu=False)

        parser.add_argument(
            "--disable_layer_norm",
            required=False,
            action="store_true",
            help="disable LayerNormalization fusion",
        )
        parser.set_defaults(disable_layer_norm=False)

        parser.add_argument(
            "--disable_gelu",
            required=False,
            action="store_true",
            help="disable Gelu fusion",
        )
        parser.set_defaults(disable_gelu=False)

        parser.add_argument(
            "--enable_gelu_approximation",
            required=False,
            action="store_true",
            help="enable Gelu/BiasGelu to FastGelu conversion",
        )
        parser.set_defaults(enable_gelu_approximation=False)

        parser.add_argument(
            "--disable_shape_inference",
            required=False,
            action="store_true",
            help="disable symbolic shape inference",
        )
        parser.set_defaults(disable_shape_inference=False)

        parser.add_argument(
            "--enable_gemm_fast_gelu",
            required=False,
            action="store_true",
            help="enable GemmfastGelu fusion",
        )
        parser.set_defaults(enable_gemm_fast_gelu=False)

        parser.add_argument(
            "--use_mask_index",
            required=False,
            action="store_true",
            help="use mask index instead of raw attention mask in attention operator",
        )
        parser.set_defaults(use_mask_index=False)

        parser.add_argument(
            "--no_attention_mask",
            required=False,
            action="store_true",
            help="no attention mask. Only works for model_type=bert",
        )
        parser.set_defaults(no_attention_mask=False)

        parser.add_argument(
            "--use_multi_head_attention",
            required=False,
            action="store_true",
            help="Use MultiHeadAttention instead of Attention operator for testing purpose. "
            "Note that MultiHeadAttention might be slower than Attention since MatMul of input projection is excluded. "
            "MultiHeadAttention has only CUDA implementation so the model can only run with cuda execution provider.",
        )
        parser.set_defaults(use_multi_head_attention=False)
