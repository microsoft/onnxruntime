// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace cuda {

template <class T, class U = float>
class QuantizeLinear final : public CudaKernel {
 public:
  QuantizeLinear(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;
};

template <class T, class U = float>
class DequantizeLinear final : public CudaKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
