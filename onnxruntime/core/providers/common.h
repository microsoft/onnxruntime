// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {

/**
Handle a potentially negative axis. Enforces negative axis is valid.
@param axis Axis to convert from negative to positive if needed.
@param tensor_rank Rank of tensor axis applies to. Tensor::Shape()::NumDimensions().
@returns non-negative axis.
*/
inline int64_t HandleNegativeAxis(int64_t axis, int64_t tensor_rank) {
  ORT_ENFORCE(axis >= -tensor_rank && axis <= tensor_rank - 1, "axis ", axis,
              " is not in valid range [-", tensor_rank, ",", tensor_rank - 1, "]");
  // Handle negative axis
  return axis = axis < 0 ? axis + tensor_rank : axis;
}

}  // namespace onnxruntime
