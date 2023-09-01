// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#include "core/framework/provider_options.h"
#include "core/common/common.h"

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
  virtual OrtStatus* SetCurrentGpuDeviceId(_In_ int device_id) = 0;
  virtual OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) = 0;

  virtual std::unique_ptr<onnxruntime::IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IAllocator> CreateCUDAPinnedAllocator(const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IDataTransfer> CreateGPUDataTransfer() = 0;

  virtual void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const double* input_data, float* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const float* input_data, double* output_data, size_t count) = 0;

  virtual Status CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) = 0;
  virtual void CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) = 0;

  virtual void CopyGpuToCpu(void* dst_ptr, const void* src_ptr, const size_t size, const OrtMemoryInfo& dst_location, const OrtMemoryInfo& src_location) = 0;
  virtual void cudaMemcpy_HostToDevice(void* dst, const void* src, size_t count) = 0;
  virtual void cudaMemcpy_DeviceToHost(void* dst, const void* src, size_t count) = 0;
  virtual int cudaGetDeviceCount() = 0;
  virtual void CUDAExecutionProviderInfo__FromProviderOptions(const onnxruntime::ProviderOptions& options, onnxruntime::CUDAExecutionProviderInfo& info) = 0;

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P) && defined(ENABLE_TRAINING)
  virtual onnxruntime::cuda::INcclService& GetINcclService() = 0;
#endif

#ifdef ENABLE_NVTX_PROFILE
  virtual void NvtxRangeCreator__BeginImpl(profile::NvtxRangeCreator* p) = 0;
  virtual void NvtxRangeCreator__EndImpl(profile::NvtxRangeCreator* p) = 0;
#endif

  virtual std::shared_ptr<onnxruntime::IExecutionProviderFactory> CreateExecutionProviderFactory(const onnxruntime::CUDAExecutionProviderInfo& info) = 0;
  virtual std::shared_ptr<onnxruntime::IAllocator> CreateCudaAllocator(int16_t device_id, size_t gpu_mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy, onnxruntime::CUDAExecutionProviderExternalAllocatorInfo& external_allocator_info, const OrtArenaCfg* default_memory_arena_cfg) = 0;

  // This function is the entry point to CUDA EP's UT cases.
  // All tests ared only called from onnxruntime_test_all.
  virtual void TestAll() {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is only implements in test code path.");
  }

 protected:
  ~ProviderInfo_CUDA() = default;  // Can only be destroyed through a subclass instance
};

}  // namespace onnxruntime
