// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <class T, class U = float>
class QuantizeLinear final : public CudaKernel {
 public:
  QuantizeLinear(const OpKernelInfo& info) : CudaKernel(info) {
    if (!info.GetAttr<int64_t>("saturate", &saturate_).IsOK()) {
      saturate_ = 1;
    }    
  }

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t saturate_;
};

template <class T, class U = float>
class DequantizeLinear final : public CudaKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
