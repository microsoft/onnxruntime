// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class AttentionBase {
 protected:
  AttentionBase(const OpKernelInfo& info);
  Status CheckInputs(const OpKernelContext* context) const;

  int num_heads_;  // number of attention heads
};

template <typename T>
class Attention : public OpKernel, public AttentionBase {
 public:
  explicit Attention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
