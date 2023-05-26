// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/reduction/reduction_ops.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {
#define JSEP_DEFINE_REDUCE_KERNEL(ReduceKernel)                                                         \
  template <typename T, bool allow_multi_axes = true>                                                   \
  class ReduceKernel : public JsKernel, public ReduceKernelBase<allow_multi_axes> {                     \
   public:                                                                                              \
    using ReduceKernelBase<allow_multi_axes>::axes_;                                                    \
    using ReduceKernelBase<allow_multi_axes>::noop_with_empty_axes_;                                    \
    using ReduceKernelBase<allow_multi_axes>::keepdims_;                                                \
    ReduceKernel(const OpKernelInfo& info) : JsKernel(info), ReduceKernelBase<allow_multi_axes>(info) { \
      JSEP_INIT_KERNEL_ATTRIBUTE(ReduceKernel, ({                                                       \
                                   "keepDims" : $1,                                                     \
                                   "noopWithEmptyAxes" : $2,                                            \
                                   "axes" : $3 ? (Array.from(HEAP32.subarray($4, $4 + $3))) : [],       \
                                 }),                                                                    \
                                 static_cast<int32_t>(keepdims_),                                       \
                                 noop_with_empty_axes_,                                                 \
                                 gsl::narrow_cast<int32_t>(axes_.size()),                               \
                                 reinterpret_cast<int32_t>(axes_.data()) >> 2);                         \
    }                                                                                                   \
  };

JSEP_DEFINE_REDUCE_KERNEL(ReduceMax);
JSEP_DEFINE_REDUCE_KERNEL(ReduceMean);
JSEP_DEFINE_REDUCE_KERNEL(ReduceMin);
JSEP_DEFINE_REDUCE_KERNEL(ReduceProd);
JSEP_DEFINE_REDUCE_KERNEL(ReduceSum);
JSEP_DEFINE_REDUCE_KERNEL(ReduceLogSum);
JSEP_DEFINE_REDUCE_KERNEL(ReduceLogSumExp);
JSEP_DEFINE_REDUCE_KERNEL(ReduceSumSquares);
}  // namespace js
}  // namespace onnxruntime
