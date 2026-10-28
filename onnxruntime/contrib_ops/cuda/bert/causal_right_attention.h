// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

template <typename T>
class CausalRightAttention final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit CausalRightAttention(const OpKernelInfo& info);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CausalRightAttention);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float scale_;
  mutable std::once_flag configure_;
};

}  // namespace onnxruntime::contrib::cuda
