// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/bert/gqa_attention_base.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class SparseAttention final : public OpKernel, public GQAAttentionBase {
 public:
  SparseAttention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int sparse_block_size_;
};

}  // namespace contrib
}  // namespace onnxruntime
