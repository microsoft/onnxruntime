// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class WebGpuExecutionProvider;
namespace webgpu {

// -----------------------------------------------------------------------
// Base class for WebGPU kernels
// -----------------------------------------------------------------------
class WebGpuKernel : public OpKernel {
 public:
  explicit WebGpuKernel(const OpKernelInfo& info);

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

  virtual Status ComputeInternal(ComputeContext& context) const = 0;

  // Override of OpKernel::PrePack for WebGPU kernels.
  // Provides a ComputeContextBase and delegates to PrePackInternal for derived class customization.
  // Note: ORT does not currently support prepacked weights in non-CPU EPs, so the prepacked_weights
  // parameter is not passed to PrePackInternal. Kernel implementations that support prepacking
  // should manage their own storage.
  Status PrePack(const Tensor& tensor,
                 int input_idx,
                 AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  // Virtual method allowing derived kernels to pre-process constant tensors.
  // Called during kernel initialization when constant tensors are available.
  // @param context The compute context providing access to WebGPU resources.
  // @param tensor The constant tensor to potentially prepack.
  // @param input_idx The index of the input being prepacked.
  // @param alloc An allocator for any memory allocations needed during prepacking.
  // @param is_packed Output parameter: set to true if the tensor was successfully prepacked,
  //                  false otherwise. The default implementation sets this to false.
  // @return Status::OK() on success, or an error status on failure.
  virtual Status PrePackInternal(ComputeContextBase& context,
                                 const Tensor& tensor,
                                 int input_idx,
                                 AllocatorPtr alloc,
                                 /*out*/ bool& is_packed);

 private:
  const WebGpuExecutionProvider& ep_;
  WebGpuContext& webgpu_context_;
};

}  // namespace webgpu
}  // namespace onnxruntime
