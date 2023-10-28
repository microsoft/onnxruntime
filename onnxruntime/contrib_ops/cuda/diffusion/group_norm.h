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
  bool use_swish_activation_;  // use SiLU (also known as Swish) activation after group normalization?
  float epsilon_;
  int num_groups_;
  bool channels_last_;
  bool has_skip_;  // true for SkipGroupNorm operator; false for GroupNorm
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
