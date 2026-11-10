// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/moe/moe_base.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <mutex>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class MoE final : public CudaKernel, public MoEBase {
 public:
  explicit MoE(const OpKernelInfo& op_kernel_info);
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  bool pack_fc1_fc3_{false};
  TensorShape fc1_fc3_shape_;
  IAllocatorUniquePtr<void> packed_fc1_fc3_;
  mutable onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler mGemmProfiler;
  mutable std::mutex mGemmProfilerMutex;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
