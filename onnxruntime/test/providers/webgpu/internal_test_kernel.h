// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/framework/config_options.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime::test {
#if defined(ORT_USE_EP_API_ADAPTERS)
using WebGpuTestOpKernel = ep::adapter::OpKernel;
using WebGpuTestOpKernelInfo = ep::adapter::OpKernelInfo;
using WebGpuTestOpKernelContext = ep::adapter::OpKernelContext;
#else
using WebGpuTestOpKernel = OpKernel;
using WebGpuTestOpKernelInfo = OpKernelInfo;
using WebGpuTestOpKernelContext = OpKernelContext;
#endif

// Supplies real host KernelInfo through the public API when testing adapter-compiled programs.
class WebGpuTestKernel {
 public:
  explicit WebGpuTestKernel(const ConfigOptions& options = {});
  ~WebGpuTestKernel();
  WebGpuExecutionProvider& Ep();
  const WebGpuTestOpKernel& Kernel();
#if defined(ORT_USE_EP_API_ADAPTERS)
  OrtKernelContext* Context();
#endif

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuTestKernel);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace onnxruntime::test
