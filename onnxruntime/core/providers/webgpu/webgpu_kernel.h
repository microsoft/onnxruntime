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

 private:
  const WebGpuExecutionProvider& ep_;
};

}  // namespace webgpu
}  // namespace onnxruntime
