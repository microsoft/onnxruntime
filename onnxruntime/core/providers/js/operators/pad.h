// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/tensor/padbase.h"

namespace onnxruntime {
namespace js {

class Pad : public JsKernel, public PadBase {
 public:
  explicit Pad(const OpKernelInfo& info) : JsKernel(info), PadBase(info) {
    std::vector<int32_t> pads;
    if (!is_dynamic_) {
      pads.resize(pads_.size());
      for (size_t i = 0; i < pads_.size(); ++i) {
        pads[i] = gsl::narrow_cast<int32_t>(pads_[i]);
      }
    }

    JSEP_INIT_KERNEL_ATTRIBUTE(Pad, ({"mode" : $1,
                                      "value" : $2,
                                      "pads" : $3 ? Array.from(HEAP32.subarray($4, $4 + $3)) : []}),
                               static_cast<int32_t>(mode_),
                               static_cast<double>(value_),
                               gsl::narrow_cast<int32_t>(pads.size()),
                               reinterpret_cast<int32_t>((pads.size() > 0) ? pads.data() : nullptr) >> 2);
  }
};

}  // namespace js
}  // namespace onnxruntime
