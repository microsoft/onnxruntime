// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
namespace onnxruntime {
class IAllocator;
class IDataTransfer;
}  // namespace onnxruntime

struct ProviderInfo_CUDA {
  virtual OrtStatus* SetCurrentGpuDeviceId(_In_ int device_id) = 0;
  virtual OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) = 0;

  virtual std::unique_ptr<onnxruntime::IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IDataTransfer> CreateGPUDataTransfer(void* stream) = 0;

  virtual void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) = 0;
  virtual void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) = 0;

  virtual bool CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) = 0;
  virtual bool CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) = 0;

  virtual void CopyGpuToCpu(void* dst_ptr, const void* src_ptr, const size_t size, const OrtMemoryInfo& dst_location, const OrtMemoryInfo& src_location) = 0;
  virtual void cudaMemcpy_HostToDevice(void* dst, const void* src, size_t count) = 0;
};

extern "C" {
#endif

/**
 * \param device_id cuda device id, starts from zero.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id);

#ifdef __cplusplus
}
#endif
