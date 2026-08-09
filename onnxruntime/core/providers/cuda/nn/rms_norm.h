// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename U, typename V>
class RMSNorm final : public CudaKernel {
 public:
  RMSNorm(const OpKernelInfo& op_kernel_info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
  double epsilon_;
  // The stash_type is not supported in the current implementation.
  // int64_t stash_type;
};

}  // namespace cuda
}  // namespace onnxruntime
