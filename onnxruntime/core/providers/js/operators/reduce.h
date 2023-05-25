// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/reduction/reduction_ops.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

template <typename T, bool allow_multi_axes = true>
class ReduceMean : public JsKernel, public ReduceKernelBase<allow_multi_axes> {
 public:
  using ReduceKernelBase<allow_multi_axes>::axes_;
  using ReduceKernelBase<allow_multi_axes>::noop_with_empty_axes_;
  using ReduceKernelBase<allow_multi_axes>::keepdims_;
  ReduceMean(const OpKernelInfo& info) : JsKernel(info), ReduceKernelBase<allow_multi_axes>(info) {
    if (axes_.empty() == true) {
      JSEP_INIT_KERNEL_ATTRIBUTE(ReduceMean, ({
                                   "keepDims" : $1,
                                   "noopWithEmptyAxes" : $2,
                                   "axes":[]
                                 }),
                                 static_cast<int32_t>(keepdims_),
                                 noop_with_empty_axes_);
    } else {
      JSEP_INIT_KERNEL_ATTRIBUTE(ReduceMean, ({
                                   "keepDims" : $1,
                                   "noopWithEmptyAxes" : $2,
                                   "axes" : (Array.from(HEAP32.subarray($4, $4 + $3))),
                                 }),
                                 static_cast<int32_t>(keepdims_),
                                 noop_with_empty_axes_,
                                 gsl::narrow_cast<int32_t>(axes_.size() ),
                                 reinterpret_cast<int32_t>(axes_.data()) >> 2);
    }
  }
};

}  // namespace js
}  // namespace onnxruntime
