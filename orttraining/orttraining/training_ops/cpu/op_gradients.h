// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <cctype>

namespace onnxruntime {
namespace contrib {

template <typename T>
class SinGrad final : public OpKernel {
 public:
  explicit SinGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SinGrad);
};

template <typename T>
class ReluGrad final : public OpKernel {
 public:
  explicit ReluGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReluGrad);
};

template <typename T>
class SoftmaxGrad final : public OpKernel {
 public:
  explicit SoftmaxGrad(const OpKernelInfo& info) : OpKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 0);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxGrad);
  int64_t axis_;
};

template <typename T>
class LogSoftmaxGrad final : public OpKernel {
 public:
  explicit LogSoftmaxGrad(const OpKernelInfo& info) : OpKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 0);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LogSoftmaxGrad);
  int64_t axis_;
};

}  // namespace contrib
}  // namespace onnxruntime
