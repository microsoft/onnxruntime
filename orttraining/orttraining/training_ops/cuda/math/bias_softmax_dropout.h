// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class BiasSoftmaxDropout final : public onnxruntime::cuda::CudaKernel {
 public:
  BiasSoftmaxDropout(const OpKernelInfo& info) : CudaKernel{info} {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(1));
    int64_t is_inner_broadcast_value;
    ORT_ENFORCE(info.GetAttr<int64_t>("is_inner_broadcast", &is_inner_broadcast_value).IsOK());
    is_inner_broadcast_ = is_inner_broadcast_value != 0;
    int64_t seed = 0;
    if (info.GetAttr<int64_t>("seed", &seed).IsOK()) {
      generator_ = std::make_unique<PhiloxGenerator>(static_cast<uint64_t>(seed));
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  // For Softmax.
  int64_t axis_;
  bool is_inner_broadcast_;

  // For Dropout.
  mutable std::unique_ptr<PhiloxGenerator> generator_;
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace cuda
}  // namespace onnxruntime
