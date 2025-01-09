// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once
#include "core/framework/tensor_shape.h"
#include "core/common/status.h"

namespace onnxruntime {

constexpr const char* kLayerNormInputShapeMismatchError =
    "Size of scale and bias (if provided) must match X.shape()[axis:], "
    "or scale and bias shape are same and can be broadcasted to X when axis is 2. ";

constexpr const char* kLayerNormInvalidSize = "Size of X.shape()[axis:] must be larger than 1, got ";

class LayerNormHelper {
 public:
  static Status CheckBroadcast(const TensorShape& x_shape,
                               const TensorShape& scale_shape,
                               const TensorShape& bias_shape,
                               bool has_bias,
                               int64_t axis,
                               int64_t& broadcast_param) {
    broadcast_param = GetBroadcastParam(x_shape, scale_shape, has_bias ? &bias_shape : nullptr, axis);
    if (broadcast_param == 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             kLayerNormInputShapeMismatchError,
                             "Shapes X=", x_shape, " scale=", scale_shape, " bias=", bias_shape, " and axis=", axis);
    }

    return Status::OK();
  }

 private:
  static int64_t GetBroadcastParam(const TensorShape& x_shape,
                                   const TensorShape& scale_shape,
                                   const TensorShape* bias_shape,
                                   int64_t axis) {
    // X shape is (B, S, ...)
    if (axis == 2 &&
        x_shape.NumDimensions() >= 3 &&
        x_shape.NumDimensions() == scale_shape.NumDimensions() &&
        (bias_shape == nullptr || *bias_shape == scale_shape)) {
      for (size_t i = 2; i < x_shape.NumDimensions(); ++i) {
        if (x_shape.GetDims()[i] != scale_shape.GetDims()[i]) {
          return 0;
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

    return 0;
  }
};

}  // namespace onnxruntime
