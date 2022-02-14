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

class BitmaskDropout final : public CudaKernel {
 public:
  BitmaskDropout(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = std::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

  static constexpr size_t NumBitmaskElements(size_t numBitElements) {
    return (numBitElements + 31) / 32;
  }

 private:
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
