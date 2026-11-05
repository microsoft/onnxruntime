# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Inference-only CPU contrib sources, matching the native CPU provider taxonomy."""

def cpu_contrib_sources():
    """Include schemas and every enabled CPU inference registration together."""
    return select({
        "//bazel:contrib_enabled": native.glob(
            [
                "onnxruntime/core/graph/contrib_ops/*.cc",
                "onnxruntime/contrib_ops/cpu/*.cc",
                "onnxruntime/contrib_ops/cpu/attnlstm/*.cc",
                "onnxruntime/contrib_ops/cpu/bert/*.cc",
                "onnxruntime/contrib_ops/cpu/math/*.cc",
                "onnxruntime/contrib_ops/cpu/moe/*.cc",
                "onnxruntime/contrib_ops/cpu/quantization/*.cc",
                "onnxruntime/contrib_ops/cpu/sparse/*.cc",
                "onnxruntime/contrib_ops/cpu/tensor/*.cc",
                "onnxruntime/contrib_ops/cpu/transformers/*.cc",
                "onnxruntime/contrib_ops/cpu/utils/*.cc",
            ],
            exclude = [
                "onnxruntime/contrib_ops/cpu/fused_activation.cc",
                # The baseline QMoE TU includes these ISA implementations itself.
                "onnxruntime/contrib_ops/cpu/moe/qmoe_shared_layout_*avx*.cc",
            ],
        ) + [
            "onnxruntime/core/optimizer/gqa_value_layout_boundaries.cc",
            "onnxruntime/core/optimizer/gqa_value_layout_transformer.cc",
        ],
        "//conditions:default": [],
    })
