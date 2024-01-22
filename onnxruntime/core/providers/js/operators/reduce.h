// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

namespace onnxruntime {
namespace js {
#define JSEP_DEFINE_REDUCE_KERNEL(ReduceKernel)                                                         \
  template <bool allow_multi_axes = true>                                                               \
  class ReduceKernel : public JsKernel, public ReduceKernelBase<allow_multi_axes> {                     \
   public:                                                                                              \
    using ReduceKernelBase<allow_multi_axes>::axes_;                                                    \
    using ReduceKernelBase<allow_multi_axes>::noop_with_empty_axes_;                                    \
    using ReduceKernelBase<allow_multi_axes>::keepdims_;                                                \
    ReduceKernel(const OpKernelInfo& info) : JsKernel(info), ReduceKernelBase<allow_multi_axes>(info) { \
      std::vector<int32_t> axes(axes_.size());                                                          \
      if (axes_.size() > 0) {                                                                           \
        std::transform(axes_.begin(), axes_.end(), axes.begin(),                                        \
                       [](int64_t axis) { return gsl::narrow_cast<int32_t>(axis); });                   \
      }                                                                                                 \
      JSEP_INIT_KERNEL_ATTRIBUTE(ReduceKernel, ({                                                       \
                                   "keepDims" : !!$1,                                                   \
                                   "noopWithEmptyAxes" : !!$2,                                          \
                                   "axes" : $3 ? (Array.from(HEAP32.subarray($3, $4))) : [],            \
                                 }),                                                                    \
                                 static_cast<int32_t>(keepdims_),                                       \
                                 static_cast<int32_t>(noop_with_empty_axes_),                           \
                                 JSEP_HEAP32_INDEX_START(axes),                                         \
                                 JSEP_HEAP32_INDEX_END(axes));                                          \
    }                                                                                                   \
  };

JSEP_DEFINE_REDUCE_KERNEL(ReduceMax);
JSEP_DEFINE_REDUCE_KERNEL(ReduceMean);
JSEP_DEFINE_REDUCE_KERNEL(ReduceMin);
JSEP_DEFINE_REDUCE_KERNEL(ReduceProd);
JSEP_DEFINE_REDUCE_KERNEL(ReduceSum);
JSEP_DEFINE_REDUCE_KERNEL(ReduceL1);
JSEP_DEFINE_REDUCE_KERNEL(ReduceL2);
JSEP_DEFINE_REDUCE_KERNEL(ReduceLogSum);
JSEP_DEFINE_REDUCE_KERNEL(ReduceLogSumExp);
JSEP_DEFINE_REDUCE_KERNEL(ReduceSumSquare);
}  // namespace js
}  // namespace onnxruntime
