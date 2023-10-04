// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

class Slice : public JsKernel, public SliceBase {
 public:
  Slice(const OpKernelInfo& info, bool dynamic = true) : JsKernel(info), SliceBase(info, dynamic) {
    auto attr_axes = AxesAttribute();
    auto attr_starts = StartsAttribute();
    auto attr_ends = EndsAttribute();
    std::vector<int32_t> axes(attr_axes.begin(), attr_axes.end());
    std::vector<int32_t> starts(attr_starts.begin(), attr_starts.end());
    std::vector<int32_t> ends(attr_ends.begin(), attr_ends.end());

    JSEP_INIT_KERNEL_ATTRIBUTE(Slice, ({"starts" : $1 ? Array.from(HEAP32.subarray($2, $2 + $1)) : [],
                                        "ends" : $3 ? Array.from(HEAP32.subarray($4, $4 + $3)) : [],
                                        "axes" : $5 ? Array.from(HEAP32.subarray($6, $6 + $5)) : []}),
                               gsl::narrow_cast<int32_t>(starts.size()),
                               reinterpret_cast<int32_t>((starts.size() > 0) ? starts.data() : nullptr) >> 2,
                               gsl::narrow_cast<int32_t>(ends.size()),
                               reinterpret_cast<int32_t>((ends.size() > 0) ? ends.data() : nullptr) >> 2,
                               gsl::narrow_cast<int32_t>(axes.size()),
                               reinterpret_cast<int32_t>((axes.size() > 0) ? axes.data() : nullptr) >> 2);
  }
};

class Slice_1 final : public Slice {
 public:
  Slice_1(const OpKernelInfo& info) : Slice(info, false) {}
};
}  // namespace js
}  // namespace onnxruntime
