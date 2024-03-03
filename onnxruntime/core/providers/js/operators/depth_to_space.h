// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

template <bool has_mode, bool is_channels_last>
class DepthToSpace final : public JsKernel {
 public:
  DepthToSpace(const OpKernelInfo& info) : JsKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("blocksize", &blocksize_).IsOK(), "Attribute blocksize is not set.");
    std::string mode = has_mode? info.GetAttrOrDefault<std::string>("mode", "DCR") : "DCR";

    if (mode != "DCR" && mode != "CRD") {
      ORT_THROW("Invalid mode attribute value: ", mode);
    }
    mode_ = std::move(mode);

    JSEP_INIT_KERNEL_ATTRIBUTE(DepthToSpace, ({
                                 "blocksize" : $1,
                                 "mode" : UTF8ToString($2),
                                 "format" : $3 ? "NHWC" : "NCHW"
                               }),
                               static_cast<int32_t>(blocksize_),
                               mode_.c_str(), static_cast<int32_t>(is_channels_last));
  }

 private:
  int64_t blocksize_;
  std::string mode_;
};

}  // namespace js
}  // namespace onnxruntime
