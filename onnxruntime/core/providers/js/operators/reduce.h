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
    if (noop_with_empty_axes_ == true) {
      JSEP_INIT_KERNEL_ATTRIBUTE(ReduceMean, ({
                                   "keepdims" : $1,
                                   "noop_with_empty_axes" : $2
                                 }),
                                 static_cast<int32_t>(keepdims_),
                                 noop_with_empty_axes_);
    } else {
      JSEP_INIT_KERNEL_ATTRIBUTE(ReduceMean, ({
                                   "axes" : ($1 ? Array.from(HEAP32.subarray($2, $2 + $1)) : []),
                                   "keepdims" : $3
                                 }),
                                 gsl::narrow_cast<int32_t>(axes_.size() ? axes_.size() : 0),
                                 reinterpret_cast<int32_t>(!axes_.empty() ? axes_.data() : nullptr) >> 2,
                                 static_cast<int32_t>(keepdims_));
    }
  }
};

}  // namespace js
}  // namespace onnxruntime
