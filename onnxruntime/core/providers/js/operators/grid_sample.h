// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

template <bool is_channels_last>
class GridSample : public JsKernel {
 public:
  GridSample(const OpKernelInfo& info) : JsKernel(info) {
    int64_t align_corners = info.GetAttrOrDefault<int64_t>("align_corners", 0);
    std::string mode = info.GetAttrOrDefault<std::string>("mode", "linear");
    std::string padding_mode = info.GetAttrOrDefault<std::string>("padding_mode", "zeros");
    int64_t channels_last = is_channels_last ? 1 : info.GetAttrOrDefault<int64_t>("channels_last", 0);

    JSEP_INIT_KERNEL_ATTRIBUTE(GridSample, ({
                                 "align_corners" : $1,
                                 "mode" : UTF8ToString($2),
                                 "padding_mode" : UTF8ToString($3),
                                 "format" : $4 ? "NHWC" : "NCHW"
                               }),
                               static_cast<int32_t>(align_corners), mode.c_str(),
                               padding_mode.c_str(), static_cast<int32_t>(channels_last));
  }
};

}  // namespace js
}  // namespace onnxruntime
