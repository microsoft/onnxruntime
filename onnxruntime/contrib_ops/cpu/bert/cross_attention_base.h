// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {

class CrossAttentionBase {
 public:
  Status CheckInputs(const Tensor* query,
                     const Tensor* key,
                     const Tensor* value,
                     const Tensor* bias,
                     void* parameters,
                     const int max_threads_per_block) const;

 protected:
  CrossAttentionBase(const OpKernelInfo& info) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);
  }

  int num_heads_;  // number of attention heads
};

}  // namespace contrib
}  // namespace onnxruntime
