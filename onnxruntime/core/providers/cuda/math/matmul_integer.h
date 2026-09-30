// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {
template <typename T1, typename T2>
class MatMulInteger final : public CudaKernel {
  using Base = CudaKernel;

 public:
  MatMulInteger(const OpKernelInfo& info) : CudaKernel(info) {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
    const auto kernel_info = info.GetKernelInfo();
    has_a_zero_point_ = info.GetInputCount() > 2 && !kernel_info.GetInputName(2).empty();
    has_b_zero_point_ = info.GetInputCount() > 3 && !kernel_info.GetInputName(3).empty();
#else
    has_a_zero_point_ = info.GetInputCount() > 2 && info.node().InputDefs()[2]->Exists();
    has_b_zero_point_ = info.GetInputCount() > 3 && info.node().InputDefs()[3]->Exists();
#endif
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool has_a_zero_point_;
  bool has_b_zero_point_;
};

}  // namespace cuda
}  // namespace onnxruntime
