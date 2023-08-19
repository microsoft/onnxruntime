// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class Cast final : public JsKernel {
 public:
  Cast(const OpKernelInfo& info) : JsKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");

    // ignore attribute 'saturate' as float8 is not supported in JSEP anyway
    JSEP_INIT_KERNEL_ATTRIBUTE(Cast, ({
                                 "to" : $1
                               }),
                               gsl::narrow_cast<int32_t>(to));
  }
};

}  // namespace js
}  // namespace onnxruntime
