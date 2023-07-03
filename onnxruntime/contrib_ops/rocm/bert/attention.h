// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/rocm/bert/attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T>
class Attention final : public RocmKernel, public AttentionBase {
 public:
  Attention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 public:
  AttentionType attn_type_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
