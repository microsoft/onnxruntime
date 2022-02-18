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

/**
 * @brief We can implement the same functionality as the standard Dropout kernel, but with a bit-packed mask.
 * 
 * Currently, only the BitmaskDropoutGrad op knows how to parse the output of this op. See "bitmask_dropout_impl.cu" for
 * details on how the bit-packing actually works for this kernel.
 * 
 * At a high level, we pack 32 booleans into a single uint32_t. The first boolean goes into the lowest bit of the first
 * uint32_t, the second boolean into the second lowest bit (and so on, up to the 32nd boolean). The 33rd boolean goes
 * into the first bit of the second uint32_t, and so on and so forth. This offers a memory efficiency improvement of
 * approximately 8x (we can use 1/8 of the previous memory for storing masks).
 */
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
