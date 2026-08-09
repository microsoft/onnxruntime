// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsKernel;

class RotaryEmbedding final : public JsKernel {
 public:
  explicit RotaryEmbedding(const OpKernelInfo& info) : JsKernel(info) {
    int64_t interleaved = info.GetAttrOrDefault<int64_t>("interleaved", 0);
    int64_t num_heads = info.GetAttrOrDefault<int64_t>("num_heads", 0);
    int64_t rotary_embedding_dim = info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0);
    float scale = info.GetAttrOrDefault<float>("scale", 1.0);

    JSEP_INIT_KERNEL_ATTRIBUTE(RotaryEmbedding, ({
                                 "interleaved" : !!$1,
                                 "numHeads" : $2,
                                 "rotaryEmbeddingDim" : $3,
                                 "scale" : $4,
                               }),
                               static_cast<int32_t>(interleaved), static_cast<int32_t>(num_heads),
                               static_cast<int32_t>(rotary_embedding_dim), scale);
  }
};

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
