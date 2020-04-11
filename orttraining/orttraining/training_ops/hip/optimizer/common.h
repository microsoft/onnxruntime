// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/hip/hip_common.h"
#include "orttraining/training_ops/cpu/optimizer/common.h"

namespace onnxruntime {
namespace hip {

template <typename T>
Status CopyIfNotSameBuffer(const Tensor& source_tensor, Tensor& target_tensor) {
  const T* source = source_tensor.template Data<T>();
  T* target = target_tensor.template MutableData<T>();
  if (target != source) {
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(target, source, source_tensor.SizeInBytes(), hipMemcpyDeviceToDevice));
  }
  return Status::OK();
}

}  // namespace hip
}  // namespace onnxruntime
