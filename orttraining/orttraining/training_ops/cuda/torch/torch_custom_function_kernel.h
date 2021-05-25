// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"
#include "core/language_interop_ops/torch/torch_proxy.h"
#include "orttraining/training_ops/cpu/torch/torch_custom_function_kernel_base.h"

namespace onnxruntime {
namespace cuda {

// Pytorch's torch.autograd.Function.apply(...) wrapper.
class PythonOp final : public contrib::PythonOpBase, public CudaKernel {
 public:
  PythonOp(const OpKernelInfo& info) : contrib::PythonOpBase(info), CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

// Pytorch's torch.autograd.Function.backward(...) wrapper.
class PythonOpGrad final : public contrib::PythonOpGradBase, public CudaKernel {
 public:
  PythonOpGrad(const OpKernelInfo& info) : contrib::PythonOpGradBase(info), CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
