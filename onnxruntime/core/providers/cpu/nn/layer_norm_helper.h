// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor_shape.h"
#include "core/common/status.h"
#include "core/common/narrow.h"

namespace onnxruntime {

constexpr const char* kLayerNormInputShapeMismatchError =
    "Size of scale and bias (if provided) must match X.shape[axis:], "
    "or scale and bias (with same shape) can be broadcasted to X when axis is 2.";

constexpr const char* kLayerNormInvalidSize = "Size of X.shape[axis:] must be larger than 1, got ";

constexpr int64_t kLayerNormInvalidInput = -1;

struct LayerNormParams {
  int64_t num_rows;
  int64_t norm_size;  // size per row
  int64_t scale_size;
  int64_t bias_size;
  int64_t broadcast_param;
};

// We support broadcasting for axis=2, where the first two dimensions are rows, and the rest are columns.
// When X shape is (B, S, ...), and x_row (index of one row in X) is in the range of [0, B * S).
// We support scale and bias shape like below:
//    When scale and bias shape is (1, 1, ...) or (...), value of broadcast_param is 0.
//    When scale and bias shape is (B, 1, ...), value of broadcast_param is S.
//    When scale and bias shape is (B, S, ...), value of broadcast_param is 1.
//    When scale and bias shape is (1, S, ...), value of broadcast_param is -S.

// Below is a macro to compute the offset for scale and bias data for a row of X.
#ifndef LAYER_NORM_SCALE_BIAS_OFFSET
#define LAYER_NORM_SCALE_BIAS_OFFSET(broadcast_param, x_row, norm_size) \
  ((broadcast_param == 0) ? 0                                           \
                          : norm_size * (broadcast_param > 0 ? x_row / broadcast_param : x_row % (-broadcast_param)))
#endif

class LayerNormHelper {
 public:
  static Status CheckInputs(const TensorShape& x_shape,
                            const TensorShape& scale_shape,
                            const TensorShape& bias_shape,
                            bool has_bias,
                            int64_t axis,
                            LayerNormParams& params) {
    params.num_rows = x_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis));
    params.norm_size = x_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis));
    params.scale_size = scale_shape.Size();
    params.bias_size = bias_shape.Size();
    params.broadcast_param = 0;

    if (params.norm_size <= 1) {
      params.broadcast_param = kLayerNormInvalidInput;
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, kLayerNormInvalidSize, params.norm_size);
    } else if (params.scale_size != params.norm_size || (has_bias && params.bias_size != params.scale_size)) {
      params.broadcast_param = GetBroadcastParam(x_shape, scale_shape, has_bias ? &bias_shape : nullptr, axis);
      if (params.broadcast_param == kLayerNormInvalidInput) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               kLayerNormInputShapeMismatchError,
                               " X.shape=", x_shape,
                               " scale.shape=", scale_shape,
                               " bias.shape=", bias_shape,
                               " and axis=", axis);
      }
    }
    return Status::OK();
  }

 private:
  static int64_t GetBroadcastParam(const TensorShape& x_shape,
                                   const TensorShape& scale_shape,
                                   const TensorShape* bias_shape,
                                   int64_t axis) {
    // Note that when size of scale and bias is norm_size, it won't enter this function (see CheckInputs).

    // X shape is (B, S, ...)
    if (axis == 2 &&
        x_shape.NumDimensions() >= 3 &&
        x_shape.NumDimensions() == scale_shape.NumDimensions() &&
        (bias_shape == nullptr || *bias_shape == scale_shape)) {
      for (size_t i = 2; i < x_shape.NumDimensions(); ++i) {
        if (x_shape.GetDims()[i] != scale_shape.GetDims()[i]) {
          // scale cannot be broadcasted to X. It is invalid input.
          return kLayerNormInvalidInput;
        }
      }

      if (x_shape.GetDims()[0] == scale_shape.GetDims()[0]) {
        // scale and bias shape is (B, S, ...).
        if (x_shape.GetDims()[1] == scale_shape.GetDims()[1]) {
          return 1;
        }

        // scale and bias shape is (B, 1, ...), returns S
        if (scale_shape.GetDims()[1] == 1) {
          return x_shape.GetDims()[1];
        }
      } else if (scale_shape.GetDims()[0] == 1) {
        // scale and bias shape is (1, S, ...), returns -S
        if (x_shape.GetDims()[1] == scale_shape.GetDims()[1]) {
          return -(x_shape.GetDims()[1]);
        }
      }
    }

    // Other cases that are not supported.
    return kLayerNormInvalidInput;
  }
};

}  // namespace onnxruntime
