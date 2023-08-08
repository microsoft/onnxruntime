// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {
class IAllocator;
class IDataTransfer;
struct IExecutionProviderFactory;
struct ROCMExecutionProviderInfo;
enum class ArenaExtendStrategy : int32_t;
struct ROCMExecutionProviderExternalAllocatorInfo;

namespace rocm {
class INcclService;
}

struct ProviderInfo_ROCM {
  virtual OrtStatus* SetCurrentGpuDeviceId(_In_ int device_id) = 0;
  virtual OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) = 0;

  virtual std::unique_ptr<onnxruntime::IAllocator> CreateROCMAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IAllocator> CreateROCMPinnedAllocator(const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IDataTransfer> CreateGPUDataTransfer() = 0;

  virtual void rocm__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) = 0;
  virtual void rocm__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) = 0;
  virtual void rocm__Impl_Cast(void* stream, const double* input_data, float* output_data, size_t count) = 0;
  virtual void rocm__Impl_Cast(void* stream, const float* input_data, double* output_data, size_t count) = 0;

  virtual Status RocmCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) = 0;
  virtual void RocmCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) = 0;

  virtual void CopyGpuToCpu(void* dst_ptr, const void* src_ptr, const size_t size, const OrtMemoryInfo& dst_location, const OrtMemoryInfo& src_location) = 0;
  virtual void rocmMemcpy_HostToDevice(void* dst, const void* src, size_t count) = 0;
  virtual void rocmMemcpy_DeviceToHost(void* dst, const void* src, size_t count) = 0;
  virtual int hipGetDeviceCount() = 0;
  virtual void ROCMExecutionProviderInfo__FromProviderOptions(const onnxruntime::ProviderOptions& options, onnxruntime::ROCMExecutionProviderInfo& info) = 0;

#if defined(USE_ROCM) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
  virtual onnxruntime::rocm::INcclService& GetINcclService() = 0;
#endif

  virtual std::shared_ptr<onnxruntime::IExecutionProviderFactory> CreateExecutionProviderFactory(const onnxruntime::ROCMExecutionProviderInfo& info) = 0;
  virtual std::shared_ptr<onnxruntime::IAllocator> CreateRocmAllocator(int16_t device_id, size_t gpu_mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy, onnxruntime::ROCMExecutionProviderExternalAllocatorInfo& external_allocator_info, OrtArenaCfg* default_memory_arena_cfg) = 0;
};

}  // namespace onnxruntime
