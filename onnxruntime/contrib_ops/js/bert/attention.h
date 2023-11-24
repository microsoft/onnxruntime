// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/attention_base.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::contrib::AttentionBase;
using onnxruntime::js::JsKernel;

class Attention : public JsKernel, AttentionBase {
 public:
  explicit Attention(const OpKernelInfo& info) : JsKernel(info), AttentionBase(info, false) {
    std::vector<int32_t> qkv_sizes(qkv_hidden_sizes_.size());
    if (qkv_hidden_sizes_.size() > 0) {
      std::transform(qkv_hidden_sizes_.begin(), qkv_hidden_sizes_.end(), qkv_sizes.begin(),
                     [](int64_t sz) { return gsl::narrow_cast<int32_t>(sz); });
    }

    JSEP_INIT_KERNEL_ATTRIBUTE(Attention, ({
                                 "numHeads" : $1,
                                 "isUnidirectional" : $2,
                                 "maskFilterValue" : $3,
                                 "scale" : $4,
                                 "doRotary" : $5,
                                 "qkvHiddenSizes" : $6 ? (Array.from(HEAP32.subarray(Number($7), Number($7) + $6))) : [],
                                 "pastPresentShareBuffer" : !!$8,
                               }),
                               static_cast<int32_t>(num_heads_),
                               static_cast<int32_t>(is_unidirectional_),
                               static_cast<int32_t>(mask_filter_value_),
                               static_cast<int32_t>(scale_),
                               static_cast<int32_t>(do_rotary_),
                               static_cast<int32_t>(qkv_hidden_sizes_.size()),
                               reinterpret_cast<uintptr_t>((qkv_sizes.size() > 0) ? qkv_sizes.data() : nullptr) >> 2,
                               static_cast<int32_t>(past_present_share_buffer_));
  }
};

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
