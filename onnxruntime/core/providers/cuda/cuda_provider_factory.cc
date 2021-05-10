// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_provider_factory_creator.h"
#include "core/providers/cuda/cuda_provider_factory.h"

#include <memory>

#include "gsl/gsl"

#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

void Shutdown_DeleteRegistry();

struct CUDAProviderFactory : IExecutionProviderFactory {
  CUDAProviderFactory(const CUDAExecutionProviderInfo& info)
      : info_{info} {}
  ~CUDAProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  CUDAExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> CUDAProviderFactory::CreateProvider() {
  return std::make_unique<CUDAExecutionProvider>(info_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(const CUDAExecutionProviderInfo& info) {
  return std::make_shared<onnxruntime::CUDAProviderFactory>(info);
}

}  // namespace onnxruntime

namespace onnxruntime {
struct ProviderInfo_CUDA_Impl : ProviderInfo_CUDA {
  OrtStatus* SetCurrentGpuDeviceId(_In_ int device_id) override {
    int num_devices;
    auto cuda_err = ::cudaGetDeviceCount(&num_devices);
    if (cuda_err != cudaSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to set device id since cudaGetDeviceCount failed.");
    }

    if (device_id >= num_devices) {
      std::ostringstream ostr;
      ostr << "Invalid device id. Device id should be less than total number of devices (" << num_devices << ")";
      return CreateStatus(ORT_INVALID_ARGUMENT, ostr.str().c_str());
    }

    cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to set device id.");
    }
    return nullptr;
  }

  OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) override {
    auto cuda_err = cudaGetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to get device id.");
    }
    return nullptr;
  }

  std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) override {
    return std::make_unique<CUDAAllocator>(device_id, name);
  }

  std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name) override {
    return std::make_unique<CUDAPinnedAllocator>(device_id, name);
  }

  std::unique_ptr<IDataTransfer> CreateGPUDataTransfer(void* stream) override {
    return std::make_unique<GPUDataTransfer>(static_cast<cudaStream_t>(stream));
  }

  void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) override {
    return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  }

  void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) override {
    return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  }

  bool CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) override { return CudaCall<cudaError, false>(cudaError(retCode), exprString, libName, cudaError(successCode), msg); }
  bool CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg) override { return CudaCall<cudaError, true>(cudaError(retCode), exprString, libName, cudaError(successCode), msg); }

  void CopyGpuToCpu(void* dst_ptr, const void* src_ptr, const size_t size, const OrtMemoryInfo& dst_location, const OrtMemoryInfo& src_location) override {
    ORT_ENFORCE(dst_location.device.Type() == OrtDevice::CPU);

    // Current CUDA device.
    int device;
    CUDA_CALL(cudaGetDevice(&device));

    if (device != src_location.id) {
      // Need to switch to the allocating device.
      CUDA_CALL(cudaSetDevice(src_location.id));
      // Copy from GPU to CPU.
      CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
      // Switch back to current device.
      CUDA_CALL(cudaSetDevice(device));
    } else {
      // Copy from GPU to CPU.
      CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
    }
  }

  // Used by slice_concatenate_test.cc and onnxruntime_pybind_state.cc
  void cudaMemcpy_HostToDevice(void* dst, const void* src, size_t count) override { CUDA_CALL_THROW(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice)); }
  // Used by onnxruntime_pybind_state.cc
  void cudaMemcpy_DeviceToHost(void* dst, const void* src, size_t count) override { CUDA_CALL_THROW(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost)); }

  int cudaGetDeviceCount() override {
    int num_devices = 0;
    CUDA_CALL_THROW(::cudaGetDeviceCount(&num_devices));
    return num_devices;
  }

  void CUDAExecutionProviderInfo__FromProviderOptions(const ProviderOptions& options, CUDAExecutionProviderInfo& info) {
    info = CUDAExecutionProviderInfo::FromProviderOptions(options);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const CUDAExecutionProviderInfo& info) override {
    return std::make_shared<CUDAProviderFactory>(info);
  }

  std::shared_ptr<IAllocator> CreateCudaAllocator(int16_t device_id, size_t gpu_mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy, onnxruntime::CUDAExecutionProviderExternalAllocatorInfo& external_allocator_info) override {
    return CUDAExecutionProvider::CreateCudaAllocator(device_id, gpu_mem_limit, arena_extend_strategy, external_allocator_info);
  }

} g_info;

struct CUDA_Provider : Provider {
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    auto params = reinterpret_cast<const OrtCUDAProviderOptions*>(void_params);

    CUDAExecutionProviderInfo info{};
    info.device_id = gsl::narrow<OrtDevice::DeviceId>(params->device_id);
    info.gpu_mem_limit = params->gpu_mem_limit;
    info.arena_extend_strategy = static_cast<onnxruntime::ArenaExtendStrategy>(params->arena_extend_strategy);
    info.cudnn_conv_algo_search = params->cudnn_conv_algo_search;
    info.do_copy_in_default_stream = params->do_copy_in_default_stream;
    info.has_user_compute_stream = params->has_user_compute_stream;
    info.user_compute_stream = params->user_compute_stream;
    info.default_memory_arena_cfg = params->default_memory_arena_cfg;

    return std::make_shared<CUDAProviderFactory>(info);
  }

  void Shutdown() override {
    Shutdown_DeleteRegistry();
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}

