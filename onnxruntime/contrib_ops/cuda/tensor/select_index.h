// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
// #include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class SelectIndex final : public ::onnxruntime::cuda::CudaKernel {
 public:
  SelectIndex(const OpKernelInfo& info) : ::onnxruntime::cuda::CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("ignore_idx", &ignore_idx_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_{0};
  int64_t ignore_idx_{0};
};
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
