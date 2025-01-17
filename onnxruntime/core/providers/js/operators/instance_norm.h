// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

template <bool is_channels_last>
class InstanceNorm : public JsKernel {
 public:
  InstanceNorm(const OpKernelInfo& info) : JsKernel(info) {
    float epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-05);
    int64_t channels_last = is_channels_last ? 1 : info.GetAttrOrDefault<int64_t>("channels_last", 0);

    JSEP_INIT_KERNEL_ATTRIBUTE(InstanceNormalization, ({
                                 "epsilon" : $1,
                                 "format" : $2 ? "NHWC" : "NCHW",
                               }),
                               static_cast<float>(epsilon_),
                               static_cast<int32_t>(channels_last));
  }
};

}  // namespace js
}  // namespace onnxruntime
