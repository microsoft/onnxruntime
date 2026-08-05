// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "contrib_ops/cpu/bert/gqa_attention_base.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsKernel;

class GroupQueryAttention : public JsKernel, GQAAttentionBase {
 public:
  explicit GroupQueryAttention(const OpKernelInfo& info)
      : JsKernel(info), GQAAttentionBase(info, false) {
    // The windowed KV cache (cache-relative indexing + shift compaction) is implemented only by the
    // CUDA kernel. Fail loudly rather than treating the window-sized buffer as a full-length cache.
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("sliding_window_cache", 0) == 0,
                "GroupQueryAttention (JS): sliding_window_cache=1 is not implemented.");
    JSEP_INIT_KERNEL_ATTRIBUTE(GroupQueryAttention, ({
                                 "numHeads" : $1,
                                 "kvNumHeads" : $2,
                                 "scale" : $3,
                                 "softcap" : $4,
                                 "doRotary" : $5,
                                 "rotaryInterleaved" : $6,
                                 "smoothSoftmax" : $7,
                                 "localWindowSize" : $8
                               }),
                               static_cast<int32_t>(num_heads_),
                               static_cast<int32_t>(kv_num_heads_),
                               static_cast<float>(scale_),
                               static_cast<float>(softcap_),
                               static_cast<int32_t>(do_rotary_),
                               static_cast<int32_t>(rotary_interleaved_),
                               static_cast<int32_t>(use_smooth_softmax_),
                               static_cast<int32_t>(local_window_size_));
  }
};

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
