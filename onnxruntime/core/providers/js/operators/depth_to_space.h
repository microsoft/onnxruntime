// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

#include <string>
#include <utility>

namespace onnxruntime {
namespace js {

template <bool is_channels_last>
class DepthToSpace final : public JsKernel {
 public:
  DepthToSpace(const OpKernelInfo& info) : JsKernel(info) {
    int64_t blocksize;
    std::string mode;
    ORT_ENFORCE(info.GetAttr<int64_t>("blocksize", &blocksize).IsOK(), "Attribute blocksize is not set.");
    mode = info.GetAttrOrDefault<std::string>("mode", "DCR");

    if (mode != "DCR" && mode != "CRD") {
      ORT_THROW("Invalid mode attribute value: ", mode);
    }

    JSEP_INIT_KERNEL_ATTRIBUTE(DepthToSpace, ({
                                 "blocksize" : $1,
                                 "mode" : UTF8ToString($2),
                                 "format" : $3 ? "NHWC" : "NCHW"
                               }),
                               static_cast<int32_t>(blocksize),
                               mode.c_str(), static_cast<int32_t>(is_channels_last));
  }
};

}  // namespace js
}  // namespace onnxruntime
