// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "opencl_utils.h"
#include "opencl_forward_decl.h"

#include "core/framework/allocator.h"
#include "core/framework/execution_provider.h"
#include "core/providers/providers.h"
#include "core/graph/constants.h"

/* Notes about OpenCL object lifetime

OpenCL have reference counted (RC) object management, however, it is impossible
to query the current counter reliably. As a result, it makes it difficualt to
*share* created object, especially of programs and kernels, which are slow to
create and resulting loooong session initialization time. So we are not relying
on the internal RC mechanism.

- cl_device_id, cl_context and cl_command_queue

  lifetime are bounded to EP

- cl_mem

  is created and managed by OpenCLBufferAllocator and OpenCLImage2DAllocator

- cl_program and cl_kernel

  are created and managed by OpenCLKernelManager

*/

namespace onnxruntime {

// Information needed to construct OpenCL execution providers.
struct OpenCLExecutionProviderInfo {
  bool use_fp16;
  int32_t auto_runing_level;
};
namespace opencl {
enum class GpuType;
struct OpenCLDeviceInfo {
  // get cache size, compute units and frequency. used to tune kernel performance
  std::string device_name;
  bool has_fp16 = false;
  int32_t gpu_model = 0;
  GpuType gpu_type;
  uint32_t sub_group_size=0;
  uint64_t global_memery_cachesize_ = 0;
  uint32_t compute_units_ = 0;
  uint32_t max_freq_ = 0;
  uint64_t local_memory_size_ = 0;
  size_t max_work_group_size = 0;
  size_t max_work_item_size[3] = {0};
  size_t image_2d_max_size[2] = {0};
};
}  // namespace opencl

using IAllocatorUniquePtrToClMem = IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>>;

// Logical device representation.
class OpenCLExecutionProvider : public IExecutionProvider {
 public:
  explicit OpenCLExecutionProvider(const OpenCLExecutionProviderInfo& info);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpenCLExecutionProvider);
  virtual ~OpenCLExecutionProvider();

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

  Status Sync() const override;
  // Status OnRunStart() override;
  // Status OnRunEnd(bool sync_stream) override;
  const void* GetExecutionHandle() const noexcept override {
    // The OpenCL interface does not return anything interesting.
    return nullptr;
  }

  /// OpenCL object accessor, kernel developer might rarely use them.
  cl_device_id GetOpenCLDevice() const { return dev_; }
  cl_context GetOpenCLContext() const { return ctx_; }
  cl_command_queue GetCommandQueue() const { return cmd_queue_; }
  cl_command_queue GetCommandQueueForTune() const { return cmd_tune_queue_; }

  // /// Get an OpenCL Buffer for temporary usage from Buffer allocator.
  cl_mem GetScratchBufferTmp(size_t nbytes) const;
  void WriteToCLBuffer(cl_mem buffer, const void* host_ptr, size_t nbytes);
  void ReadFromCLBuffer(cl_mem buffer,void* host_ptr,size_t nbytes);

  void ReleaseCLBuffer(cl_mem buffer) {
      if (buffer == NULL) {
          return;
      }

      cl_int err;
      err = clReleaseMemObject(buffer);
      if (err != CL_SUCCESS) {
          printf("Error releasing buffer: %d\n", err);
      }
  }

  // /// Get an OpenCL Buffer for temporary usage from Image2D allocator.
  // IAllocatorUniquePtrToClMem GetScratchImage2D(const opencl::Image2DDesc& desc) const;

  // Utility for other classes, kernel developer should not call them directly.
  // They are left as public to avoid friending those classes.

  /// Whether fp16 is enabled. Used by Image2D allocator to decide the datatype
  /// and OpenCLProgramManager to build kernels.

  bool UseFp16() const { return use_fp16_; }
  int32_t TuneEnabledLevel() const { return auto_runing_level_; }
  /// OpenCL after kernel launch performance heuristic. This is called in
  /// KernelLauncher
  Status AfterCLLaunch() const;

  const std::string GetCompileOptions() const { return compile_options_; };

  /// OpenCLKernel use it to initialize OpenCLKernelHolder
  const opencl::OpenCLProgramManager& GetProgramManager() const;
  opencl::OpenCLProgramManager& GetProgramManager();

  opencl::NDRange DefaultLocalWG2DOrTune(const opencl::NDRange& gws, const opencl::TuneKernelWithTimeFunc& func) const;

 private:
  Status InitCompileOptions();
  Status InitOpenCLContext();
  void DisableFp16() { use_fp16_ = false; }

  cl_device_id dev_;
  cl_context ctx_;
  cl_command_queue cmd_queue_;
  cl_command_queue cmd_tune_queue_;
  int32_t auto_runing_level_;
  bool use_fp16_;
  bool flush_after_launch_;
  std::string compile_options_;
  opencl::OpenCLDeviceInfo dev_info_;

 private:
  std::unique_ptr<opencl::OpenCLProgramManager> program_manager_;

  // IDataTransfer is a lightweight interface with std::unique_ptr as its
  // return value. Bind kernels to it directly will cause the kernel being
  // created from time to time. So we move the kernels here.
  std::unique_ptr<opencl::OpenCLKernelHolder> copy_kernels_;
  void InitCopyKernels();

#ifdef TRACY_ENABLE
  TracyCLCtx tracy_cl_ctx_;

 public:
  TracyCLCtx GetTracyCLContext() { return tracy_cl_ctx_; }
  const std::remove_pointer_t<TracyCLCtx>* GetTracyCLContext() const { return tracy_cl_ctx_; }
#endif
};

}  // namespace onnxruntime
