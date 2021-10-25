// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "opencl_utils.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/providers/providers.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct OpenCL execution providers.
struct OpenCLExecutionProviderInfo {
  OpenCLExecutionProviderInfo() = default;
};

// Logical device representation.
class OpenCLExecutionProvider : public IExecutionProvider {
 public:
  explicit OpenCLExecutionProvider(const OpenCLExecutionProviderInfo& info);
  OpenCLExecutionProvider(OpenCLExecutionProvider&&) noexcept;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OpenCLExecutionProvider);
  virtual ~OpenCLExecutionProvider();

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  cl::Device GetOpenCLDevice() const { return dev_; }
  cl::Context GetOpenCLContext() const { return ctx_; }
  cl::CommandQueue GetCommandQueue() const { return cmd_queue_; }

 private:
  Status InitOpenCLContext();
  Status InitOpenCLAllocator();

  cl::Device dev_;
  cl::Context ctx_;
  cl::CommandQueue cmd_queue_;
};

}  // namespace onnxruntime
