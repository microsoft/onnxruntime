// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class GroupNorm final : public CudaKernel {
 public:
  GroupNorm(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool use_swish_activation_;
  float epsilon_;
  int num_groups_;
  bool channels_last_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
