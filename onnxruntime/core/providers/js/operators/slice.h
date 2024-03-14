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

    JSEP_INIT_KERNEL_ATTRIBUTE(Slice, ({"starts" : $1 ? Array.from(HEAP32.subarray($1, $2)) : [],
                                        "ends" : $3 ? Array.from(HEAP32.subarray($3, $4)) : [],
                                        "axes" : $5 ? Array.from(HEAP32.subarray($5, $6)) : []}),
                               JSEP_HEAP32_INDEX_START(starts),
                               JSEP_HEAP32_INDEX_END(starts),
                               JSEP_HEAP32_INDEX_START(ends),
                               JSEP_HEAP32_INDEX_END(ends),
                               JSEP_HEAP32_INDEX_START(axes),
                               JSEP_HEAP32_INDEX_END(axes));
  }
};

class Slice_1 final : public Slice {
 public:
  Slice_1(const OpKernelInfo& info) : Slice(info, false) {}
};
}  // namespace js
}  // namespace onnxruntime
