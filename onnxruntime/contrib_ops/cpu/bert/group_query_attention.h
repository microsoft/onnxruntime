// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "gqa_attention_base.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class GroupQueryAttention final : public OpKernel, public GQAAttentionBase {
 public:
  GroupQueryAttention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
