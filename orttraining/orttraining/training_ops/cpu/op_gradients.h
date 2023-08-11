// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include <cctype>

namespace onnxruntime {
namespace contrib {

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
class QuickGeluGrad final : public OpKernel {
 public:
  explicit QuickGeluGrad(const OpKernelInfo& info) : OpKernel(info) {
    alpha_ = info.GetAttrOrDefault<float>("alpha", 1.702f);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QuickGeluGrad);
  float alpha_;
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
    opset_ = (node.OpType() == "SoftmaxGrad_13" || node.OpType() == "LogSoftmaxGrad_13") ? 13 : 1;
    axis_ = info.GetAttrOrDefault("axis", static_cast<int64_t>(opset_ < 13 ? 1 : -1));
    is_logsoftmaxgrad_ = node.OpType() == "LogSoftmaxGrad_13" || node.OpType() == "LogSoftmaxGrad";
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SoftmaxGrad);
  int64_t axis_;
  int opset_;  // opset_ of the forward Softmax operator
  bool is_logsoftmaxgrad_;
};

template <typename T>
class LeakyReluGrad final : public OpKernel {
 public:
  explicit LeakyReluGrad(const OpKernelInfo& info) : OpKernel(info) {
    alpha_ = info.GetAttrOrDefault<float>("alpha", 0.01f);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LeakyReluGrad);
  float alpha_;
};

}  // namespace contrib
}  // namespace onnxruntime
