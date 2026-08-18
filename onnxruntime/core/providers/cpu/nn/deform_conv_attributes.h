// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

// Shared attributes for ONNX DeformConv (opset 19+).
// See https://onnx.ai/onnx/operators/onnx__DeformConv.html
// Used by both CPU and CUDA implementations (CUDA includes from here).
struct DeformConvAttributes {
  template <typename KernelInfoType>
  explicit DeformConvAttributes(const KernelInfoType& info) {
    // Optional attributes.
    // If not present, they will be empty/default, and handled in Compute/ComputeInternal.
    (void)info.GetAttrs("kernel_shape", kernel_shape);
    (void)info.GetAttrs("strides", strides);
    (void)info.GetAttrs("pads", pads);
    (void)info.GetAttrs("dilations", dilations);
    group = info.template GetAttrOrDefault<int64_t>("group", 1);
    offset_group = info.template GetAttrOrDefault<int64_t>("offset_group", 1);
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

// Common derived dimensions used by both CPU and CUDA kernels.
struct DeformConvCommonDims {
  int64_t kernel_size{0};        // kH * kW
  int64_t output_image_size{0};  // out_h * out_w
  int64_t input_image_size{0};   // H * W_in
  int64_t kernel_dim{0};         // (C / group) * kernel_size
};

// Validates shared runtime bounds and computes common derived dimensions.
// This helper is backend-agnostic and intended to be reused by both CPU/CUDA
// after DeformConvValidateAndParse() succeeds.
inline Status DeformConvValidateAndComputeCommonDims(const DeformConvParams& params,
                                                     DeformConvCommonDims& dims) {
  const int64_t int64_max = std::numeric_limits<int64_t>::max();
  ORT_RETURN_IF_NOT(params.N > 0 && params.C > 0 && params.M > 0 &&
                        params.group > 0 && params.offset_group > 0 &&
                        params.kH > 0 && params.kW > 0 &&
                        params.H > 0 && params.W_in > 0 &&
                        params.out_h > 0 && params.out_w > 0,
                    "Invalid deform conv dimensions.");

  ORT_RETURN_IF_NOT(params.kH <= int64_max / params.kW, "kernel_size overflows int64.");
  dims.kernel_size = params.kH * params.kW;

  ORT_RETURN_IF_NOT(params.out_h <= int64_max / params.out_w, "output_image_size overflows int64.");
  dims.output_image_size = params.out_h * params.out_w;

  ORT_RETURN_IF_NOT(params.H <= int64_max / params.W_in, "input_image_size overflows int64.");
  dims.input_image_size = params.H * params.W_in;

  ORT_RETURN_IF_NOT((params.C / params.group) <= int64_max / dims.kernel_size, "kernel_dim overflows int64.");
  dims.kernel_dim = (params.C / params.group) * dims.kernel_size;

  return Status::OK();
}

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
  ORT_RETURN_IF_NOT(params.N >= 0, "Batch size N must be non-negative.");
  ORT_RETURN_IF_NOT(params.C > 0, "Input channels C must be positive.");
  ORT_RETURN_IF_NOT(params.M > 0, "Output channels M (oC) must be positive.");
  ORT_RETURN_IF_NOT(W_shape[1] > 0, "Weight W must have positive in-channels (W_shape[1] = C/group).");

  // Handle kernel shape inference. If kernel_shape is provided, it must match weight spatial dims
  // to avoid GEMM using wrong K and potential out-of-bounds reads from the weight buffer.
  const int64_t W_kH = W_shape[2];
  const int64_t W_kW = W_shape[3];
  if (!attrs.kernel_shape.empty()) {
    ORT_RETURN_IF_NOT(attrs.kernel_shape.size() == 2,
                      "kernel_shape must be absent or have exactly 2 values (kH, kW) for 2D DeformConv.");
    ORT_RETURN_IF_NOT(attrs.kernel_shape[0] == W_kH && attrs.kernel_shape[1] == W_kW,
                      "kernel_shape must match weight spatial dimensions (W_shape[2], W_shape[3]).");
    params.kH = attrs.kernel_shape[0];
    params.kW = attrs.kernel_shape[1];
  } else {
    params.kH = W_kH;
    params.kW = W_kW;
  }

  // DeformConv is 2D-only: when an attribute is present, require exact length to avoid silently misinterpreting malformed models.
  params.pad_h = params.pad_w = params.pad_h_end = params.pad_w_end = 0;
  if (!attrs.pads.empty()) {
    ORT_RETURN_IF_NOT(attrs.pads.size() == 4,
                      "pads must be absent or have exactly 4 values [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end] for 2D DeformConv.");
    params.pad_h = attrs.pads[0];
    params.pad_w = attrs.pads[1];
    params.pad_h_end = attrs.pads[2];
    params.pad_w_end = attrs.pads[3];
    ORT_RETURN_IF_NOT(params.pad_h >= 0 && params.pad_w >= 0 && params.pad_h_end >= 0 && params.pad_w_end >= 0,
                      "Pads must be non-negative (ONNX spec).");
  }

  if (!attrs.strides.empty()) {
    ORT_RETURN_IF_NOT(attrs.strides.size() == 2,
                      "strides must be absent or have exactly 2 values [stride_h, stride_w] for 2D DeformConv.");
    params.stride_h = attrs.strides[0];
    params.stride_w = attrs.strides[1];
  } else {
    params.stride_h = params.stride_w = 1;
  }

  if (!attrs.dilations.empty()) {
    ORT_RETURN_IF_NOT(attrs.dilations.size() == 2,
                      "dilations must be absent or have exactly 2 values [dilation_h, dilation_w] for 2D DeformConv.");
    params.dilation_h = attrs.dilations[0];
    params.dilation_w = attrs.dilations[1];
  } else {
    params.dilation_h = params.dilation_w = 1;
  }
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

  // CPU BilinearInterpolate uses int for indices (for performance optimization); W <= int_max / (H+1) covers all index math.
  ORT_RETURN_IF_NOT(params.H >= 0 && params.W_in >= 0, "Input spatial dimensions H and W must be non-negative.");
  ORT_RETURN_IF_NOT(params.W_in <= static_cast<int64_t>(std::numeric_limits<int>::max()) / (params.H + 1),
                    "Input (H+1)*W must not exceed int max (for performance optimization).");

  // Validate tensor shapes (use division to avoid int64 overflow in offset_group * 2 * kH * kW).
  ORT_RETURN_IF_NOT(offset_shape[0] == params.N, "Offset batch size must match input batch size.");
  const int64_t offset_block = 2 * params.kH * params.kW;
  ORT_RETURN_IF_NOT(offset_block > 0 && offset_shape[1] % offset_block == 0 &&
                        offset_shape[1] / offset_block == params.offset_group,
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
    const int64_t mask_block = params.kH * params.kW;
    ORT_RETURN_IF_NOT(mask_block > 0 && (*mask_shape)[1] % mask_block == 0 &&
                          (*mask_shape)[1] / mask_block == params.offset_group,
                      "Mask channel count must be offset_group * kH * kW.");
    ORT_RETURN_IF_NOT((*mask_shape)[2] == params.out_h, "Mask spatial height must match output oH.");
    ORT_RETURN_IF_NOT((*mask_shape)[3] == params.out_w, "Mask spatial width must match output oW.");
  }

  return Status::OK();
}

}  // namespace onnxruntime
