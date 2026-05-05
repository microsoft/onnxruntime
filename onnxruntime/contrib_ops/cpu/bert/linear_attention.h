// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include <string>

namespace onnxruntime {
namespace contrib {

template <typename T>
class LinearAttention final : public OpKernel {
 public:
  LinearAttention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int q_num_heads_;
  int kv_num_heads_;
  std::string update_rule_;
  float scale_;
  int chunk_size_;
};

}  // namespace contrib
}  // namespace onnxruntime
