// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

template <bool is_channels_last>
class BatchNorm final : public JsKernel {
 public:
  explicit BatchNorm(const OpKernelInfo& info) : JsKernel(info) {
    float epsilon = info.GetAttrOrDefault<float>("epsilon", 1e-5);
    float momentum = info.GetAttrOrDefault<float>("momentum", 0.9);
    int64_t spatial = info.GetAttrOrDefault<int64_t>("spatial", 1);

    const auto& node = info.node();
    int opset = node.SinceVersion();
    int64_t training_mode = opset <= 9 ? info.GetOutputCount() > 1 : info.GetAttrOrDefault<int64_t>("training_mode", 0);

    JSEP_INIT_KERNEL_ATTRIBUTE(BatchNormalization, ({
                                 "epsilon" : $1,
                                 "momentum" : $2,
                                 "spatial" : !!$4,
                                 "trainingMode" : !!$3,
                                 "format" : $5 ? "NHWC" : "NCHW",
                               }),
                               static_cast<float>(epsilon), static_cast<float>(momentum),
                               static_cast<int32_t>(training_mode), static_cast<int32_t>(spatial),
                               static_cast<int32_t>(is_channels_last));
  }
};

}  // namespace js
}  // namespace onnxruntime
