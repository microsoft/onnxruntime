// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cpu/bert/longformer_attention_base.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class QOrderedLongformerAttention final : public CudaKernel, public LongformerAttentionBase {
 public:
  QOrderedLongformerAttention(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool use_compact_memory_;
  int order_input_;
  int order_weight_;
  int order_global_weight_;
  int order_output_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
