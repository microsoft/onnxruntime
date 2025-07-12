// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename U>
class GroupNorm final : public CudaKernel {
 public:
  GroupNorm(const OpKernelInfo& op_kernel_info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  double epsilon_;
  int64_t num_groups_;
  int64_t stash_type_;
};

}  // namespace cuda
}  // namespace onnxruntime