// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class TrainableDropoutCurand final : public CudaKernel {
 public:
  TrainableDropoutCurand(const OpKernelInfo& info) : CudaKernel(info), default_ratio_(0.5) {
    // TODO: what is default value for seed?
    info.GetAttrOrDefault<int64_t>("seed", &seed_, 0);
    CURAND_CALL_THROW(curandSetPseudoRandomGeneratorSeed(CurandGenerator(), seed_));
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t seed_;
  const float default_ratio_;
};

template <typename T>
class TrainableDropoutGradCurand final : public CudaKernel {
 public:
  TrainableDropoutGradCurand(const OpKernelInfo& info) : CudaKernel(info), default_ratio_(0.5) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  const float default_ratio_;
};
}  // namespace cuda
}  // namespace onnxruntime
