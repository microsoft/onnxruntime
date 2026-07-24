// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class PagedAttention final : public OpKernel {
 public:
  explicit PagedAttention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int num_heads_;
  int kv_num_heads_;
  bool do_rotary_;
  float scale_;
  float softcap_;
};

}  // namespace contrib
}  // namespace onnxruntime