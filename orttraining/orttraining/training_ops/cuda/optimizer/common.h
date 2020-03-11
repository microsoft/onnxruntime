// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status CopyIfNotSameBuffer(const Tensor& source_tensor, Tensor& target_tensor) {
  const T* source = source_tensor.template Data<T>();
  T* target = target_tensor.template MutableData<T>();
  if (target != source) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target, source, source_tensor.SizeInBytes(), cudaMemcpyDeviceToDevice));
  }
  return Status::OK();
}

template <typename TS, typename TC>
TC compute_bias_correction_coefficient(
  const TC momentum_update_coefficient,
  const TS step) {
  if (step > 0) {
    return TC(1.0 - std::pow(static_cast<double>(momentum_update_coefficient), static_cast<double>(step)));
  } else {
    return TC(1.f);
  }
}

}  // namespace cuda
}  // namespace onnxruntime
