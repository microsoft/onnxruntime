// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"
#include <core/common/safeint.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using onnxruntime::OpKernelContext;
using onnxruntime::OpKernelInfo;
using onnxruntime::cuda::CudaKernel;
class UnfoldTensor final : public CudaKernel {
 public:
  UnfoldTensor(const OpKernelInfo& info) : CudaKernel(info) {
    dim_ = SafeInt<int>(info.GetAttrOrDefault<int64_t>("dim", -1LL));
    step_ = SafeInt<int>(info.GetAttrOrDefault<int64_t>("step", 1LL));
    ORT_ENFORCE(step_ > 0, "step must greater than zero!");

    int64_t temp_size;
    ORT_ENFORCE(info.GetAttr("size", &temp_size).IsOK());
    size_ = SafeInt<int>(temp_size);
  }

  ~UnfoldTensor() = default;

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int dim_;
  int size_;
  int step_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
