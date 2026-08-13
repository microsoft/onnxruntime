// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"

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
  MLAS_LINEAR_ATTENTION_RULE rule_;  // update_rule_, resolved once in the constructor
  float scale_;
  int chunk_size_;
  // Always 0 on CPU (a state window is CUDA-only), but kept so the shared shape helper in
  // linear_attention_helper.h is driven the same way on every EP.
  int state_window_;
};

}  // namespace contrib
}  // namespace onnxruntime
