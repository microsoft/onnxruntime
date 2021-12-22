// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "opencl_utils.h"
#include "opencl_forward_decl.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/providers/providers.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct OpenCL execution providers.
struct OpenCLExecutionProviderInfo {
  bool use_fp16;
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

  cl_device_id GetOpenCLDevice() const { return dev_; }
  cl_context GetOpenCLContext() const { return ctx_; }
  cl_command_queue GetCommandQueue() const { return cmd_queue_; }

  IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>> GetScratchBuffer(size_t nbytes) const;
  IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>> GetScratchImage2D(opencl::Image2DDesc desc) const;

  bool UseFp16() const { return use_fp16_; }

  Status AfterCLLaunch() const;

 private:
  Status InitOpenCLContext();
  void DisableFp16() { use_fp16_ = false; }
  static bool ShouldFlushAfterLaunch(const std::string& device_name);

  cl_device_id dev_;
  cl_context ctx_;
  cl_command_queue cmd_queue_;
  bool use_fp16_;
  bool flush_after_launch_;

 private:
  // IDataTransfer is a lightweight interface with std::unique_ptr as its
  // return value. Bind kernels to it directly will cause the kernel being
  // created from time to time. So we move the kernels here.
  std::unique_ptr<opencl::OpenCLKernelHolder> copy_kernels_;
  void InitCopyKernels();
};

}  // namespace onnxruntime
