// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class GroupQueryAttention final : public CudaKernel {
 public:
  GroupQueryAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  int num_heads_;     // number of attention heads
  int kv_num_heads_;  // different for k and v for group query attention
  bool left_padding_; // shifts last token to end of buffer
  bool is_unidirectional_;  // causal
  bool is_past_bsnh_;
  float scale_;
  bool disable_flash_attention_;
  bool disable_memory_efficient_attention_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
