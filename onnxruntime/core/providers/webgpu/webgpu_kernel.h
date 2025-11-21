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

  Status PrePack(const Tensor& tensor,
                 int input_idx,
                 AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

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
