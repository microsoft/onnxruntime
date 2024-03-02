// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class FastGelu final : public CudaKernel {
 public:
  FastGelu(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
#ifndef USE_ROCM
  bool use_half2_;
#endif
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
