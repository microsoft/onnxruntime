// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <climits>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

// Shared attributes for ONNX DeformConv (opset 19+).
// See https://onnx.ai/onnx/operators/onnx__DeformConv.html
// Used by both CPU and CUDA implementations (CUDA includes from here).
struct DeformConvAttributes {
  explicit DeformConvAttributes(const OpKernelInfo& info) {
    // Optional attributes.
    // If not present, they will be empty/default, and handled in Compute/ComputeInternal.
    (void)info.GetAttrs("kernel_shape", kernel_shape);
    (void)info.GetAttrs("strides", strides);
    (void)info.GetAttrs("pads", pads);
    (void)info.GetAttrs("dilations", dilations);
    group = info.GetAttrOrDefault<int64_t>("group", 1);
    offset_group = info.GetAttrOrDefault<int64_t>("offset_group", 1);
  }

  TensorShapeVector kernel_shape;
  TensorShapeVector strides;
  TensorShapeVector pads;
  TensorShapeVector dilations;
  int64_t group{1};
  int64_t offset_group{1};
};

// Parsed and validated parameters from DeformConv inputs.
// Used by both CPU and CUDA implementations.
// Field names align with ONNX DeformConv spec: https://onnx.ai/onnx/operators/onnx__DeformConv.html
struct DeformConvParams {
  // Input X shape (N, C, H, W)
  int64_t N{0};     // Batch size
  int64_t C{0};     // Number of input channels
  int64_t H{0};     // Input height
  int64_t W_in{0};  // Input width (W_in to avoid collision with weight W)

  // Weight W shape (oC, C/group, kH, kW)
  int64_t M{0};   // Number of output channels (oC)
  int64_t kH{0};  // Kernel height
  int64_t kW{0};  // Kernel width

  // Pads [x1_begin, x2_begin, x1_end, x2_end] for spatial axes H, W
  int64_t pad_h{0};
  int64_t pad_w{0};
  int64_t pad_h_end{0};
  int64_t pad_w_end{0};

  // Strides and dilations along each spatial axis (default 1)
  int64_t stride_h{1};
  int64_t stride_w{1};
  int64_t dilation_h{1};
  int64_t dilation_w{1};

  // Attributes: C and oC must be divisible by group; C must be divisible by offset_group
  int64_t group{1};         // Number of groups for input/output channels
  int64_t offset_group{1};  // Number of groups of offset

  // Output Y shape (N, oC, oH, oW)
  int64_t out_h{0};  // Output height (oH)
  int64_t out_w{0};  // Output width (oW)

  bool use_mask{false};  // Whether optional mask input is provided
};

// Validates inputs and parses attributes into params.
// Returns Status::OK() on success; on failure, params may be partially filled.
inline Status DeformConvValidateAndParse(
    const DeformConvAttributes& attrs,
    const TensorShape& X_shape,
    const TensorShape& W_shape,
    const TensorShape& offset_shape,
    const TensorShape* B_shape,
    const TensorShape* mask_shape,
    DeformConvParams& params) {
  ORT_RETURN_IF_NOT(X_shape.NumDimensions() == 4, "Input X must be 4D (N, C, H, W).");
  ORT_RETURN_IF_NOT(W_shape.NumDimensions() == 4, "Weight must be 4D.");
  ORT_RETURN_IF_NOT(offset_shape.NumDimensions() == 4, "Offset must be 4D.");

  // Parse input shapes
  params.N = X_shape[0];
  params.C = X_shape[1];
  params.H = X_shape[2];
  params.W_in = X_shape[3];
  params.M = W_shape[0];

  // Handle kernel shape inference
  params.kH = attrs.kernel_shape.size() >= 1 ? attrs.kernel_shape[0] : W_shape[2];
  params.kW = attrs.kernel_shape.size() >= 2 ? attrs.kernel_shape[1] : W_shape[3];

  params.pad_h = params.pad_w = params.pad_h_end = params.pad_w_end = 0;
  if (attrs.pads.size() >= 4) {
    params.pad_h = attrs.pads[0];
    params.pad_w = attrs.pads[1];
    params.pad_h_end = attrs.pads[2];
    params.pad_w_end = attrs.pads[3];
  }

  params.stride_h = attrs.strides.empty() ? 1 : attrs.strides[0];
  params.stride_w = attrs.strides.size() < 2 ? 1 : attrs.strides[1];
  params.dilation_h = attrs.dilations.empty() ? 1 : attrs.dilations[0];
  params.dilation_w = attrs.dilations.size() < 2 ? 1 : attrs.dilations[1];
  params.group = attrs.group;
  params.offset_group = attrs.offset_group;
  params.use_mask = (mask_shape != nullptr);

  // Validate attributes
  ORT_RETURN_IF_NOT(params.stride_h > 0 && params.stride_w > 0, "Strides must be positive.");
  ORT_RETURN_IF_NOT(params.dilation_h > 0 && params.dilation_w > 0, "Dilations must be positive.");
  ORT_RETURN_IF_NOT(params.kH > 0 && params.kW > 0, "Kernel shape must be positive.");
  ORT_RETURN_IF_NOT(params.group > 0, "group must be positive");
  ORT_RETURN_IF_NOT(params.offset_group > 0, "offset_group must be positive");

  params.out_h = (params.H + params.pad_h + params.pad_h_end - params.dilation_h * (params.kH - 1) - 1) / params.stride_h + 1;
  params.out_w = (params.W_in + params.pad_w + params.pad_w_end - params.dilation_w * (params.kW - 1) - 1) / params.stride_w + 1;
  ORT_RETURN_IF_NOT(params.out_h >= 0 && params.out_w >= 0, "Computed output spatial size must be non-negative.");

  // CPU BilinearInterpolate uses int for indices (for performance optimization); W <= INT_MAX / (H+1) covers all index math.
  ORT_RETURN_IF_NOT(params.H >= 0 && params.W_in >= 0, "Input spatial dimensions H and W must be non-negative.");
  ORT_RETURN_IF_NOT(params.W_in <= static_cast<int64_t>(INT_MAX) / (params.H + 1),
                    "Input (H+1)*W must not exceed INT_MAX (for performance optimization).");

  // Validate tensor shapes
  ORT_RETURN_IF_NOT(offset_shape[0] == params.N, "Offset batch size must match input batch size.");
  ORT_RETURN_IF_NOT(
      offset_shape[1] == params.offset_group * 2 * params.kH * params.kW,
      "Offset channel count must be offset_group * 2 * kH * kW.");
  ORT_RETURN_IF_NOT(offset_shape[2] == params.out_h, "Offset spatial height must match output oH.");
  ORT_RETURN_IF_NOT(offset_shape[3] == params.out_w, "Offset spatial width must match output oW.");
  ORT_RETURN_IF_NOT(params.C % params.offset_group == 0, "Input channels must be divisible by offset_group.");
  ORT_RETURN_IF_NOT(params.C == W_shape[1] * params.group, "Input channels must match weight in channels * group.");
  ORT_RETURN_IF_NOT(params.M % params.group == 0, "Output channels must be divisible by group.");

  if (B_shape != nullptr) {
    ORT_RETURN_IF_NOT(B_shape->NumDimensions() == 1, "Bias B must be 1D.");
    ORT_RETURN_IF_NOT((*B_shape)[0] == params.M, "Bias B must have shape [M] (M = number of output channels).");
  }

  // Validate mask if present
  if (params.use_mask) {
    ORT_RETURN_IF_NOT(mask_shape->NumDimensions() == 4, "Mask must be 4D.");
    ORT_RETURN_IF_NOT((*mask_shape)[0] == params.N, "Mask batch size must match input batch size.");
    ORT_RETURN_IF_NOT(
        (*mask_shape)[1] == params.offset_group * params.kH * params.kW,
        "Mask channel count must be offset_group * kH * kW.");
    ORT_RETURN_IF_NOT((*mask_shape)[2] == params.out_h, "Mask spatial height must match output oH.");
    ORT_RETURN_IF_NOT((*mask_shape)[3] == params.out_w, "Mask spatial width must match output oW.");
  }

  return Status::OK();
}

}  // namespace onnxruntime
