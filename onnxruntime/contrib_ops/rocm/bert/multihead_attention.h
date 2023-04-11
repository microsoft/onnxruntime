// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/providers/rocm/rocm_kernel.h"
#include "contrib_ops/rocm/bert/attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T>
class MultiHeadAttention final : public RocmKernel {
 public:
  MultiHeadAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  int num_heads_;  // number of attention heads
  float mask_filter_value_;
  float scale_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
