// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/framework/random_generator.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void BiasDropoutKernelImpl(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    const int64_t N,
    const fast_divmod fdm_dim,
    const float ratio,
    PhiloxGenerator& generator,
    const T* X_data,
    const T* bias_data,
    const T* residual_data,
    T* Y_data,
    bool* mask_data);

class BiasDropout final : public CudaKernel {
 public:
  BiasDropout(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = std::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
