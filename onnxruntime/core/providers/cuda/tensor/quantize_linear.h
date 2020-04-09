// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace cuda {

template <class T>
class QuantizeLinear final : public CudaKernel {
 public:
  QuantizeLinear(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;
};

template <class T>
class DequantizeLinear final : public CudaKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* p_op_kernel_context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
