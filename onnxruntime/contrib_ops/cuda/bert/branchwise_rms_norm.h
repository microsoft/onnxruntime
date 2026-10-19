// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

template <typename T>
class BranchwiseRMSNorm final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit BranchwiseRMSNorm(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
  int64_t num_branches_;
  bool use_onnx_arithmetic_;
  int64_t input_start_;
  int64_t input_width_;
};

}  // namespace onnxruntime::contrib::cuda
