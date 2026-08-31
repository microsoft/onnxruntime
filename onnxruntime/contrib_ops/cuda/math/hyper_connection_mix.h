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
class HyperConnectionMix final : public CudaKernel {
 public:
  HyperConnectionMix(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int sinkhorn_iterations_;
  float epsilon_;
  float hc_epsilon_;
  float sinkhorn_epsilon_;
  float post_alpha_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
