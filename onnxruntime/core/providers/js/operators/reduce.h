// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/reduction/reduction_ops.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

template <typename T, bool allow_multi_axes = true>
class ReduceKernel : public JsKernel, public ReduceKernelBase<allow_multi_axes> {
 public:
  using ReduceKernelBase<allow_multi_axes>::axes_;
  using ReduceKernelBase<allow_multi_axes>::noop_with_empty_axes_;
  using ReduceKernelBase<allow_multi_axes>::keepdims_;
  ReduceKernel(const OpKernelInfo& info) : JsKernel(info), ReduceKernelBase<allow_multi_axes>(info) {
    JSEP_INIT_KERNEL_ATTRIBUTE(ReduceMean, ({
                                 "keepDims" : $1,
                                 "noopWithEmptyAxes" : $2,
                                 "axes" : $3 ? (Array.from(HEAP32.subarray($4, $4 + $3))) : [],
                               }),
                               static_cast<int32_t>(keepdims_),
                               noop_with_empty_axes_,
                               gsl::narrow_cast<int32_t>(axes_.size()),
                               reinterpret_cast<int32_t>(axes_.data()) >> 2);
  }
};

template <typename T>
class ReduceMean final : public ReduceKernel<T, true> {
 public:
  ReduceMean(const OpKernelInfo& info) : ReduceKernel<T, true>(info) {
  }
};

template <typename T>
class ReduceMax final : public ReduceKernel<T, true> {
 public:
  ReduceMax(const OpKernelInfo& info) : ReduceKernel<T, true>(info) {
  }
};

template <typename T>
class ReduceMin final : public ReduceKernel<T, true> {
 public:
  ReduceMin(const OpKernelInfo& info) : ReduceKernel<T, true>(info) {
  }
};

template <typename T>
class ReduceProd final : public ReduceKernel<T, true> {
 public:
  ReduceProd(const OpKernelInfo& info) : ReduceKernel<T, true>(info) {
  }
};

template <typename T>
class ReduceSum final : public ReduceKernel<T, true> {
 public:
  ReduceSum(const OpKernelInfo& info) : ReduceKernel<T, true>(info) {
  }
};

template <typename T>
class ReduceLogSum final : public ReduceKernel<T, true> {
 public:
  ReduceLogSum(const OpKernelInfo& info) : ReduceKernel<T, true>(info) {
  }
};

}  // namespace js
}  // namespace onnxruntime
