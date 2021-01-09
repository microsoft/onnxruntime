// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
template <typename T1, typename T2>
class MatMulInteger final : public CudaKernel {
  using Base = CudaKernel;

 public:
  MatMulInteger(const OpKernelInfo& info) : CudaKernel(info) {
    has_a_zero_point_ = false;
    has_b_zero_point_ = false;
    if (info.GetInputCount() > 2) {
      has_a_zero_point_ = true;
    }
    if (info.GetInputCount() > 3) {
      has_b_zero_point_ = true;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool has_a_zero_point_;
  bool has_b_zero_point_;
};

}  // namespace cuda
}  // namespace onnxruntime
