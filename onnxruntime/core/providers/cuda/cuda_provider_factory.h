// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {
class IAllocator;
class IDataTransfer;
struct IExecutionProviderFactory;
struct CUDAExecutionProviderInfo;
enum class ArenaExtendStrategy : int32_t;
struct CUDAExecutionProviderExternalAllocatorInfo;

namespace cuda {
class INcclService;
}
namespace profile {
class NvtxRangeCreator;
}

struct ProviderInfo_CUDA {
  virtual ~ProviderInfo_CUDA() {} // This is declared due to a TSA warning, the only instantiation of this class is a global variable of automatic storage.

  virtual OrtStatus* SetCurrentGpuDeviceId(_In_ int device_id) = 0;
  virtual OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) = 0;

  virtual std::unique_ptr<onnxruntime::IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IDataTransfer> CreateGPUDataTransfer(void* stream) = 0;

  virtual void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const double* input_data, float* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const float* input_data, double* output_data, size_t count) = 0;

  virtual Status CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) = 0;
  virtual void CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) = 0;

  virtual void CopyGpuToCpu(void* dst_ptr, const void* src_ptr, const size_t size, const OrtMemoryInfo& dst_location, const OrtMemoryInfo& src_location) = 0;
  virtual void cudaMemcpy_HostToDevice(void* dst, const void* src, size_t count) = 0;
  virtual void cudaMemcpy_DeviceToHost(void* dst, const void* src, size_t count) = 0;
  virtual int cudaGetDeviceCount() = 0;
  virtual void CUDAExecutionProviderInfo__FromProviderOptions(const onnxruntime::ProviderOptions& options, onnxruntime::CUDAExecutionProviderInfo& info) = 0;

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
  virtual onnxruntime::cuda::INcclService& GetINcclService() = 0;
#endif

#ifdef ENABLE_NVTX_PROFILE
  virtual void NvtxRangeCreator__BeginImpl(profile::NvtxRangeCreator* p) = 0;
  virtual void NvtxRangeCreator__EndImpl(profile::NvtxRangeCreator* p) = 0;
#endif

  virtual std::shared_ptr<onnxruntime::IExecutionProviderFactory> CreateExecutionProviderFactory(const onnxruntime::CUDAExecutionProviderInfo& info) = 0;
  virtual std::shared_ptr<onnxruntime::IAllocator> CreateCudaAllocator(int16_t device_id, size_t gpu_mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy, onnxruntime::CUDAExecutionProviderExternalAllocatorInfo& external_allocator_info, OrtArenaCfg* default_memory_arena_cfg) = 0;

#ifndef NDEBUG
  // This function is the entry point to CUDA EP's internal (aka not accessible from bridge code for shared library)
  // tests and is only called from onnxruntime_test_all. Release builds don't need this function.
  virtual bool TestAll() = 0;
#endif
};

}  // namespace onnxruntime
