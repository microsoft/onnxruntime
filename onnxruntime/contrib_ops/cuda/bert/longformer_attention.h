// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/longformer_attention_base.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class LongformerAttention final : public CudaKernel, public LongformerAttentionBase {
 public:
  LongformerAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool use_compact_memory_;
  bool use_half4_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
