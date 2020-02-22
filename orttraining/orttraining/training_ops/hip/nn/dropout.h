// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/hip/hip_common.h"
#include "orttraining/training_ops/hip/nn/dropout_impl.h"

namespace onnxruntime {
namespace hip {

template <typename T>
class TrainableDropout final : public HipKernel {
 public:
  TrainableDropout(const OpKernelInfo& info) : HipKernel(info), default_ratio_(0.5) {
    int64_t seed = 0;
    int64_t default_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    info.GetAttrOrDefault<int64_t>("seed", &seed, default_seed);

    // TODO(bahuang): Seed is currently fixed for convergence verification purpose, will revert
    //generator_.SetSeed(static_cast<uint64_t>(seed));
    generator_.SetSeed(static_cast<uint64_t>(42));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable DropoutGenerator generator_;
  const float default_ratio_;
};

template <typename T>
class TrainableDropoutGrad final : public HipKernel {
 public:
  TrainableDropoutGrad(const OpKernelInfo& info) : HipKernel(info), default_ratio_(0.5) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  const float default_ratio_;
};

}  // namespace hip
}  // namespace onnxruntime
