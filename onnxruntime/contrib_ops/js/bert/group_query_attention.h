// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsKernel;

class GroupQueryAttention : public JsKernel {
 public:
  explicit GroupQueryAttention(const OpKernelInfo& info)
      : JsKernel(info) {
    int64_t num_heads = 0;
    int64_t kv_num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
    num_heads_ = static_cast<int>(num_heads);
    kv_num_heads_ = static_cast<int>(kv_num_heads);
    scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
    JSEP_INIT_KERNEL_ATTRIBUTE(GroupQueryAttention, ({
                                 "numHeads" : $1,
                                 "kvNumHeads" : $2,
                                 "scale" : $3,
                               }),
                               static_cast<int32_t>(num_heads_),
                               static_cast<int32_t>(kv_num_heads_),
                               static_cast<float>(scale_));
  }

 protected:
  int num_heads_;     // number of attention heads
  int kv_num_heads_;  // number of k and v heads
  float scale_;       // custom scale will be used if specified. Default value is 1/sqrt(head_size)
};

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
