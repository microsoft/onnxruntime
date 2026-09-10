// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/moe/moe_base.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"
#include "core/common/common.h"
#include "nccl_kernels.h"

#include <mutex>

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

using namespace onnxruntime::cuda;

template <typename T>
class ShardedMoE final : public NcclKernel, public MoEBase {
 public:
  explicit ShardedMoE(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t local_experts_start_index_;
  int64_t tensor_shards_;

  mutable onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler mGemmProfiler;
  mutable std::mutex mGemmProfilerMutex;
};

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
