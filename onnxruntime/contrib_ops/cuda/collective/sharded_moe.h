// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// ShardedMoE still targets the ft_moe CutlassMoeFCRunner, which was removed when MoE moved to
// llm/moe_gemm. Opt in only once it has been ported; otherwise an NCCL build cannot compile.
#if defined(ORT_USE_NCCL) && defined(ORT_ENABLE_SHARDED_MOE)
#include "contrib_ops/cuda/moe/ft_moe/moe_kernel.h"
#endif
#include "contrib_ops/cuda/moe/moe_base.h"
#include "core/common/common.h"
#include "nccl_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL) && defined(ORT_ENABLE_SHARDED_MOE)

using namespace onnxruntime::cuda;

template <typename T>
class ShardedMoE final : public NcclKernel, public MoEBase {
 public:
  explicit ShardedMoE(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  Status SynchronizeExpertsStartIndex(AllocatorPtr& alloc) const;

  int64_t local_experts_start_index_;
  int64_t tensor_shards_;
  InlinedVector<int64_t> rank_to_experts_start_index_;
};

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
