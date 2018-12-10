// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/upsample.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Upsample : public UpsampleBase<T>, public CudaKernel {
 public:
  Upsample(OpKernelInfo info) : UpsampleBase<T>(info), CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
