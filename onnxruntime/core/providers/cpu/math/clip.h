// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {

namespace clip_internal {

template <typename T>
class Clip_6Base {
 public:
  explicit Clip_6Base(const OpKernelInfo& info) {
    auto min_val = std::numeric_limits<T>::lowest();
    auto max_val = std::numeric_limits<T>::max();
    info.GetAttrOrDefault("min", &min_, min_val);
    info.GetAttrOrDefault("max", &max_, max_val);
    ORT_ENFORCE(min_ <= max_);
  }

 protected:
  T max_;
  T min_;
};
}  // namespace clip_internal

template <typename T>
class Clip_6 final : public clip_internal::Clip_6Base<T>, public OpKernel {
 public:
  explicit Clip_6(const OpKernelInfo& info) : clip_internal::Clip_6Base<T>(info), OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override;
};

// Since version 11. Min and Max are inputs
// version 12 adds type support
class Clip final : public OpKernel {
 public:
  explicit Clip(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  template <typename T>
  struct ComputeImpl;
};

}  // namespace onnxruntime
