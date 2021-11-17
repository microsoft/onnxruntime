// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "opencl_utils.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/providers/providers.h"
#include "core/graph/constants.h"

#include "opencl_forward_decl.h"
#include "opencl_data_transfer.h"

namespace onnxruntime {

// Information needed to construct OpenCL execution providers.
struct OpenCLExecutionProviderInfo {
  OpenCLExecutionProviderInfo() = default;
};

// Logical device representation.
class OpenCLExecutionProvider : public IExecutionProvider {
  friend class opencl::OpenCLDataTransfer;
 public:

  explicit OpenCLExecutionProvider(const OpenCLExecutionProviderInfo& info);
  OpenCLExecutionProvider(OpenCLExecutionProvider&&) noexcept;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OpenCLExecutionProvider);
  virtual ~OpenCLExecutionProvider();

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
  void RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) override;

  cl::Device GetOpenCLDevice() const { return dev_; }
  cl::Context GetOpenCLContext() const { return ctx_; }
  cl::CommandQueue GetCommandQueue() const { return cmd_queue_; }

 private:
  Status InitOpenCLContext();

  cl::Device dev_;
  cl::Context ctx_;
  cl::CommandQueue cmd_queue_;

 private:
  // IDataTransfer is a lightweight interface with std::unique_ptr as its
  // return value. Bind kernels to it directly will cause the kernel being
  // created from time to time. So we move the kernels here.
  void InitKernelsForDataTransfer();
  const cl::Kernel& GetCopyBuffer1DToImage2DKernel() const;
  const cl::Kernel& GetCopyImage2DToBuffer1DKernel() const;

  cl::Program program_copy_1d_;
  cl::Kernel kernel_copy_btoi_;
  cl::Kernel kernel_copy_itob_;
};

}  // namespace onnxruntime
