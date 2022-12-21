// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class FakeQuant final : public CudaKernel {
 public:
  FakeQuant(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault("quant_min", &quant_min_, static_cast<decltype(quant_min_)>(0));
    info.GetAttrOrDefault("quant_max", &quant_max_, static_cast<decltype(quant_max_)>(255));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t quant_min_;
  int64_t quant_max_;
};

template <typename T>
class FakeQuantGrad final : public CudaKernel {
 public:
  FakeQuantGrad(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
