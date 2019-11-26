// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"

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

/**
Returns true if given tensor is a scalar or 1D tensor of size 1
**/
inline bool IsScalarOr1ElementVector(const Tensor* input) {
  if (input->Shape().NumDimensions() == 0 ||
      (input->Shape().NumDimensions() == 1 && input->Shape().GetDims().size() == 1)) {
    return true;
  } else {
    return false;
  }
}

/**
Clamps input between provided min and max values
**/
inline float clamp(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

/**
Tries to call the given function parallelly, splitted into specified batch(es)
**/
template <typename F>
inline void TryBatchParallelFor(concurrency::ThreadPool* tp, int32_t total, F&& fn, int32_t batch_size = 0) {
  if (tp != nullptr) {
    if (batch_size <= 0) {
      batch_size = NumThreads() + 1;
    }
    tp->BatchParallelFor(total, batch_size, fn);
  } else {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int32_t i = 0; i < total; ++i) {
      fn(i);
    }
  }
}

/**
Tries to call the given function parallelly
**/
template <typename F>
inline void TryParallelFor(concurrency::ThreadPool* tp, int32_t total, F&& fn) {
  if (tp != nullptr) {
    tp->ParallelFor(total, fn);
  } else {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int32_t i = 0; i < total; ++i) {
      fn(i);
    }
  }
}

}  // namespace onnxruntime
