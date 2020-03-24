// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "common.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class SGDOptimizer final : public OpKernel {
 public:
  SGDOptimizer(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class AdamOptimizer final : public OpKernel {
 public:
  AdamOptimizer(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault("alpha", &alpha_, 0.9f);
    info.GetAttrOrDefault("beta", &beta_, 0.999f);
    info.GetAttrOrDefault("lambda", &lambda_, 0.0f);
    info.GetAttrOrDefault("epsilon", &epsilon_, 1e-8f);
    info.GetAttrOrDefault("do_bias_correction", &do_bias_correction_, static_cast<int64_t>(1));
    ORT_ENFORCE(alpha_ >= 0);
    ORT_ENFORCE(beta_ >= 0);
    ORT_ENFORCE(lambda_ >= 0);
    ORT_ENFORCE(epsilon_ >= 0);
    ORT_ENFORCE(do_bias_correction_ == 0 || do_bias_correction_ == 1);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
  float lambda_;
  float epsilon_;
  int64_t do_bias_correction_;
};
}  // namespace contrib
}  // namespace onnxruntime
