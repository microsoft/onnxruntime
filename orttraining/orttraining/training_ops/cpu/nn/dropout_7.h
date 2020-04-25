// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include <random>
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class Dropout_7 final : public OpKernel {
 public:
  Dropout_7(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<float>("ratio", &ratio_).IsOK());
    keep_prob_ = 1.0f - ratio_;

    // TODO: enable following when is_train is present
    /*int64_t is_train = 1;
      ORT_ENFORCE(info.GetAttr("is_train", &is_train).IsOK());
      is_train_ = (is_train == 1);*/
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool is_train_ = false;
  float ratio_;
  float keep_prob_;
};
}  // namespace onnxruntime
