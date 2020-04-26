// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/training_ops/cpu/optimizer/common.h"
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
    ORT_ENFORCE(alpha_ >= 0);
    ORT_ENFORCE(beta_ >= 0);
    ORT_ENFORCE(lambda_ >= 0);
    ORT_ENFORCE(epsilon_ >= 0);
    int64_t tmp_flag = static_cast<int64_t>(0);
    ORT_ENFORCE(info.GetAttr<int64_t>("do_bias_correction", &tmp_flag).IsOK(), "Missing/Invalid do_bias_correction");
    ORT_ENFORCE(tmp_flag == 0 || tmp_flag == 1, "do_bias_correction must be either 0 or 1.");
    do_bias_correction_ = tmp_flag != 0 ? true : false;
    info.GetAttrOrDefault("weight_decay_mode", &weight_decay_mode_, static_cast<int64_t>(0));
    ORT_ENFORCE(weight_decay_mode_ == 0 || weight_decay_mode_ == 1, "Only 0 and 1 are supported for weight decay mode.");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
  float lambda_;
  float epsilon_;
  bool do_bias_correction_;
  int64_t weight_decay_mode_;
};
}  // namespace contrib
}  // namespace onnxruntime
