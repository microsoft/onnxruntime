// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

template <typename T>
class HyperConnectionPreMix final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit HyperConnectionPreMix(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t num_branches_;
  float reduction_scale_;
};

}  // namespace onnxruntime::contrib::cuda
