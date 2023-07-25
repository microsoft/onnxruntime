// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

namespace onnxruntime {
namespace js {
#define JSEP_DEFINE_ARGMINMAX_KERNEL(ArgMinMaxKernel)                                                        \
  template <typename T, bool allow_multi_axes = false>                                                        \
  class ArgMinMaxKernel : public JsKernel, public ReduceKernelBase<allow_multi_axes> {                       \
   public:                                                                                                   \
    using ReduceKernelBase<allow_multi_axes>::axes_;                                                         \
    using ReduceKernelBase<allow_multi_axes>::select_last_index_;                                            \
    using ReduceKernelBase<allow_multi_axes>::keepdims_;                                                     \
    ArgMinMaxKernel(const OpKernelInfo& info) : JsKernel(info), ReduceKernelBase<allow_multi_axes>(info) {   \
      std::vector<int32_t> axes(axes_.size());                                                               \
      if (axes_.size() > 0) {                                                                                \
        std::transform(axes_.begin(), axes_.end(), axes.begin(),                                             \
                       [](int64_t axis) { return gsl::narrow_cast<int32_t>(axis); });                        \
      }                                                                                                      \
      JSEP_INIT_KERNEL_ATTRIBUTE(ArgMinMaxKernel, ({                                                         \
                                   "keepDims" : !!$1,                                                        \
                                   "select_last_index" : !!$2,                                               \
                                   "axes" : $3 ? (Array.from(HEAP32.subarray($4, $4 + $3))) : [],            \
                                 }),                                                                         \
                                 static_cast<int32_t>(keepdims_),                                            \
                                 static_cast<int32_t>(select_last_index_),                                   \
                                 gsl::narrow_cast<int32_t>(axes.size()),                                     \
                                 reinterpret_cast<int32_t>((axes.size() > 0) ? axes.data() : nullptr) >> 2); \
    }                                                                                                        \
  };

JSEP_DEFINE_ARGMINMAX_KERNEL(ArgMax);
JSEP_DEFINE_ARGMINMAX_KERNEL(ArgMin);
}  // namespace js
}  // namespace onnxruntime
