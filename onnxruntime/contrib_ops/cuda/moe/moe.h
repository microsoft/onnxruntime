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
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  mutable onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler mGemmProfiler;
  mutable std::mutex mGemmProfilerMutex;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
