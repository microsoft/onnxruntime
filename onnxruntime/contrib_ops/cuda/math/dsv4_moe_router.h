// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// DeepSeek-V4's MoE routing decision, from the gate GEMM's output to the two tensors QMoE
// needs: the log-domain router row for this rank's expert columns, and the local weight mass
// that has to be multiplied back in afterwards.
template <typename T>
class DSV4MoERouter final : public CudaKernel {
 public:
  explicit DSV4MoERouter(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t topk_;
  int64_t local_expert_start_;
  int64_t local_expert_count_;
  float route_scale_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
