// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// This file contains some helper functions that are used for implementing ONNX type/shape inference.

namespace ONNX_NAMESPACE {
struct InferenceContext;
}

namespace onnxruntime {
namespace contrib {
void AttentionTypeAndShapeInference(ONNX_NAMESPACE::InferenceContext& ctx, int past_input_index);
void EmbedLayerNormalizationShapeInference(::ONNX_NAMESPACE::InferenceContext& ctx);
void SkipLayerNormalizationShapeInference(::ONNX_NAMESPACE::InferenceContext& ctx);
}  // namespace contrib
}  // namespace onnxruntime
