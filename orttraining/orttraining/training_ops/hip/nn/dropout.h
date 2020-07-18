// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/hip/hip_common.h"
#include "orttraining/training_ops/hip/nn/dropout_impl.h"

namespace onnxruntime {
namespace hip {

class DropoutGrad final : public HipKernel {
 public:
  DropoutGrad(const OpKernelInfo& info) : HipKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  static constexpr float default_ratio_ = 0.5f;
};

class BiasDropout final : public HipKernel {
 public:
  BiasDropout(const OpKernelInfo& info) : HipKernel(info) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = onnxruntime::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace hip
}  // namespace onnxruntime
