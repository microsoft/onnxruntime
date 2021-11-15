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
class SigmoidGrad final : public OpKernel {
 public:
  explicit SigmoidGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SigmoidGrad);
};

template <typename T>
class TanhGrad final : public OpKernel {
 public:
  explicit TanhGrad(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TanhGrad);
};

template <typename T>
class SoftmaxGrad final : public OpKernel {
 public:
  explicit SoftmaxGrad(const OpKernelInfo& info) : OpKernel(info) {
    const auto& node = info.node();
    opset_ = node.SinceVersion();
    axis_ = info.GetAttrOrDefault("axis", static_cast<int64_t>(opset_ < 13 ? 1 : -1));
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxGrad);
  int64_t axis_;
  int opset_;
};

template <typename T>
class LogSoftmaxGrad final : public OpKernel {
 public:
  explicit LogSoftmaxGrad(const OpKernelInfo& info) : OpKernel(info) {
    const auto& node = info.node();
    opset_ = node.SinceVersion();
    axis_ = info.GetAttrOrDefault("axis", static_cast<int64_t>(opset_ < 13 ? 1 : -1));
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LogSoftmaxGrad);
  int64_t axis_;
  int opset_;
};

}  // namespace contrib
}  // namespace onnxruntime
