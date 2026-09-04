// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "gqa_attention_base.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class GroupQueryAttention : public OpKernel, public GQAAttentionBase {
 public:
  GroupQueryAttention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  // sliding_window_cache attribute: the past/present KV buffers are window-sized and the kernel
  // works in cache-relative coordinates, evicting from the front. Requires local_window_size_ > 0.
  bool sliding_window_cache_;
};

// CPU-only OSCAR mixed-precision variant. INT2 PER_GROUP quantization is inherent to this operator,
// and the sink/recent high-precision window sizes come from node attributes instead of session
// configuration. It reuses GroupQueryAttention::Compute; the derived constructor forces the OSCAR
// quantization configuration after the base constructor has parsed the shared attributes.
template <typename T>
class MixedPrecisionGroupQueryAttention final : public GroupQueryAttention<T> {
 public:
  explicit MixedPrecisionGroupQueryAttention(const OpKernelInfo& info);
};

}  // namespace contrib
}  // namespace onnxruntime
