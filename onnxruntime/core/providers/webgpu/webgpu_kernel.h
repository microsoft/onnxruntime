// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/compute_context.h"

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace webgpu {

// -----------------------------------------------------------------------
// Base class for WebGPU kernels
// -----------------------------------------------------------------------
class WebGpuKernel : public OpKernel {
 public:
  explicit WebGpuKernel(const OpKernelInfo& info)
      : OpKernel(info) {
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    ComputeContext context{*p_op_kernel_context};

    context.PushErrorScope();
    Status s = ComputeInternal(context);
    ORT_RETURN_IF_ERROR(context.PopErrorScope());

    return s;
  }

  virtual Status ComputeInternal(ComputeContext& context) const = 0;
};

}  // namespace webgpu
}  // namespace onnxruntime
