// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/moe/ft_moe/moe_kernel.h"
#include "contrib_ops/cuda/moe/moe_base.h"
#include "core/common/common.h"
#include "nccl_kernels.h"

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
};

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
