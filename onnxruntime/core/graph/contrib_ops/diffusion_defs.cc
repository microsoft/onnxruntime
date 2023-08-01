// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/onnx_function_util.h"
#include "core/graph/contrib_ops/shape_inference_functions.h"

// Suppress a warning: global initializer calls a non-constexpr function 'symbol' which is from
// ONNX_OPERATOR_SET_SCHEMA_EX macro and only happens in debug build
#if defined(_WIN32) && !defined(NDEBUG)
#pragma warning(disable : 26426)
#endif

namespace onnxruntime {
namespace contrib {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::TensorShapeProto;
#ifndef NDEBUG
using ONNX_NAMESPACE::DbgOperatorSetTracker;
#endif

constexpr const char* GroupNorm_ver1_doc = R"DOC(
Applies Group Normalization over a mini-batch of inputs as described in the paper Group Normalization (https://arxiv.org/abs/1803.08494).

This operator transforms input according to
  y = gamma * (x - mean) / sqrt(variance + epsilon) + beta

The input channels are separated into num_groups groups, each containing num_channels / num_groups channels. num_channels must be divisible by num_groups. The mean and standard-deviation are calculated separately over the each group.
The weight and bias are per-channel affine transform parameter vectors of size num_channels.

The activation attribute can be used to enable activation after group normalization.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    GroupNorm, 1,
    OpSchema()
        .SetDoc(GroupNorm_ver1_doc)
        .Attr("epsilon", "The epsilon value to use to avoid division by zero", AttributeProto::FLOAT, static_cast<float>(1e-5))
        .Attr("groups",
              "The number of groups of channels. It should be a divisor of the number of channels C",
              AttributeProto::INT)
        .Attr("activation",
              "Activation after group normalization: 0 for None, 1 for Swish",
              AttributeProto::INT)
        .Attr("channels_last",
              "1 if the input and output are in the NHWC layout, 0 if it is in the NCHW layout. Defaults to 1.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .Input(0,
               "X",
               "Input data tensor. Dimensions are (N x H x W x C) when channels_last is 1 or (N x C x H x W) otherwise, where N is the batch size, C is the number of channels, and H and W are the height and width of the data",
               "T")
        .Input(1,
               "gamma",
               "1D gamma tensor for normalization with shape (C), where C is number of channels",
               "M")
        .Input(2,
               "beta",
               "1D beta tensor for normalization  with shape (C), where C is number of channels",
               "M")
        .Output(0,
                "Y",
                "The output tensor of the same shape as X",
                "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input X and output Y types to float tensors.")
        .TypeConstraint("M", {"tensor(float16)", "tensor(float)"}, "Constrain gamma and beta to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* BiasSplitGelu_ver1_doc = R"DOC(
A fusion used in diffusion model that after adding bias, hidden state is sliced into two tensors of same size, then left
tensor multiplies the Gelu activation result of right tensor.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    BiasSplitGelu, 1,
    OpSchema()
        .SetDoc(BiasSplitGelu_ver1_doc)
        .Input(0,
               "X",
               "Input tensor. Dimensions are (N, S, D), where N is the batch size, S are image size, and D is hidden dimension",
               "T")
        .Input(1,
               "bias",
               "Bias tensor. Dimensions are (D), where D is the same hidden dimension as input tensor",
               "T")
        .Output(0,
                "Y",
                "The output tensor with dimensions (N, S, D/2)",
                "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input X and output Y types to float tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasInputShape(ctx, 0) && hasInputShape(ctx, 1)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim().size() != 3) {
              fail_shape_inference("input shall be 3 dimensions");
            }

            auto& bias_shape = getInputShape(ctx, 1);
            if (bias_shape.dim().size() != 1) {
              fail_shape_inference("bias shall be 1 dimension");
            }

            TensorShapeProto output_shape;
            *output_shape.add_dim() = input_shape.dim(0);
            *output_shape.add_dim() = input_shape.dim(1);
            if (bias_shape.dim(0).has_dim_value()) {
              output_shape.add_dim()->set_dim_value(bias_shape.dim(0).dim_value() / 2);
            } else {
              output_shape.add_dim();
            }

            updateOutputShape(ctx, 0, output_shape);
          }
        }));

constexpr const char* BiasAdd_ver1_doc = R"DOC(
Add input with bias, then add residual inputs.
)DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    BiasAdd, 1,
    OpSchema()
        .SetDoc(BiasAdd_ver1_doc)
        .Input(0,
               "X",
               "Input tensor. Dimensions are (N, S, C), where N is the batch size, S is image size H*W, and C is number of channels",
               "T")
        .Input(1,
               "bias",
               "Bias tensor. Dimensions are (C)",
               "T")
        .Input(2,
               "skip",
               "Residual tensor. Dimensions are (N, S, C)",
               "T")
        .Output(0,
                "Y",
                "The output tensor with dimensions (N, S, C)",
                "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));
}  // namespace contrib
}  // namespace onnxruntime
