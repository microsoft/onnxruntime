// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cann/cann_kernel.h"

namespace onnxruntime {
namespace cann {

class Dropout final : public CannKernel {
 public:
  Dropout(const OpKernelInfo& info) : CannKernel(info) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      seed_ = seed;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t seed_;
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace cann
}  // namespace onnxruntime
