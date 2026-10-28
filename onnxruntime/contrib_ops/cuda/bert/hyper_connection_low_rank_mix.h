// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

template <typename T>
class HyperConnectionLowRankMix final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit HyperConnectionLowRankMix(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t num_branches_;
  float epsilon_;
  float reduction_scale_;
  float silu_alpha_;
  float gate_div_;
  float gate_mul_;
  bool weights_are_constant_{false};
};

}  // namespace onnxruntime::contrib::cuda
