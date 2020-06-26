// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "attention_cpu_base.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class Attention : public OpKernel, public AttentionCPUBase {
 public:
  explicit Attention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
