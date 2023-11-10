// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_provider_factory_creator.h"
#include "core/providers/cuda/cuda_provider_options.h"

#include <memory>
#include <chrono>

#include "core/common/gsl.h"

#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

#ifdef ENABLE_NVTX_PROFILE
#include "nvtx_profile.h"
#endif

using namespace onnxruntime;

namespace onnxruntime {

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P) && defined(ENABLE_TRAINING)
namespace cuda {
cuda::INcclService& GetINcclService();
}
#endif

void InitializeRegistry();
void DeleteRegistry();

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

struct ProviderInfo_CUDA_Impl final : ProviderInfo_CUDA {
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

  std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(const char* name) override {
    return std::make_unique<CUDAPinnedAllocator>(name);
  }

  std::unique_ptr<IDataTransfer> CreateGPUDataTransfer() override {
    return std::make_unique<GPUDataTransfer>();
  }

  void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) override {
    return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  }

  void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) override {
    return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  }

  void cuda__Impl_Cast(void* stream, const double* input_data, float* output_data, size_t count) override {
    return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  }

  void cuda__Impl_Cast(void* stream, const float* input_data, double* output_data, size_t count) override {
    return cuda::Impl_Cast(static_cast<cudaStream_t>(stream), input_data, output_data, count);
  }

  Status CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { return CudaCall<cudaError, false>(cudaError(retCode), exprString, libName, cudaError(successCode), msg, file, line); }
  void CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { CudaCall<cudaError, true>(cudaError(retCode), exprString, libName, cudaError(successCode), msg, file, line); }

  void CopyGpuToCpu(void* dst_ptr, const void* src_ptr, const size_t size, const OrtMemoryInfo& dst_location, const OrtMemoryInfo& src_location) override {
    ORT_ENFORCE(dst_location.device.Type() == OrtDevice::CPU);

    // Current CUDA device.
    int device;
    CUDA_CALL_THROW(cudaGetDevice(&device));

    if (device != src_location.id) {
      // Need to switch to the allocating device.
      CUDA_CALL_THROW(cudaSetDevice(src_location.id));
      // Copy from GPU to CPU.
      CUDA_CALL_THROW(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
      // Switch back to current device.
      CUDA_CALL_THROW(cudaSetDevice(device));
    } else {
      // Copy from GPU to CPU.
      CUDA_CALL_THROW(cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost));
    }
  }

  // Used by slice_concatenate_test.cc and onnxruntime_pybind_state.cc

  void cudaMemcpy_HostToDevice(void* dst, const void* src, size_t count) override {
    // cudaMemcpy() operates on the default stream
    CUDA_CALL_THROW(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));

    // To ensure that the copy has completed, invoke a stream sync for the default stream.
    // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
    // For transfers from pageable host memory to device memory, a stream sync is performed before the copy is initiated.
    // The function will return once the pageable buffer has been copied to the staging memory for DMA transfer
    // to device memory, but the DMA to final destination may not have completed.

    CUDA_CALL_THROW(cudaStreamSynchronize(0));
  }

  // Used by onnxruntime_pybind_state.cc
  void cudaMemcpy_DeviceToHost(void* dst, const void* src, size_t count) override {
    // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync
    // For transfers from device to either pageable or pinned host memory, the function returns only once the copy has completed.
    CUDA_CALL_THROW(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
  }

  int cudaGetDeviceCount() override {
    int num_devices = 0;
    CUDA_CALL_THROW(::cudaGetDeviceCount(&num_devices));
    return num_devices;
  }

  void CUDAExecutionProviderInfo__FromProviderOptions(const ProviderOptions& options, CUDAExecutionProviderInfo& info) override {
    info = CUDAExecutionProviderInfo::FromProviderOptions(options);
  }

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P) && defined(ENABLE_TRAINING)
  cuda::INcclService& GetINcclService() override {
    return cuda::GetINcclService();
  }
#endif

#ifdef ENABLE_NVTX_PROFILE
  void NvtxRangeCreator__BeginImpl(profile::NvtxRangeCreator* p) override { p->BeginImpl(); }
  void NvtxRangeCreator__EndImpl(profile::NvtxRangeCreator* p) override { p->EndImpl(); }
#endif

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const CUDAExecutionProviderInfo& info) override {
    return std::make_shared<CUDAProviderFactory>(info);
  }

  std::shared_ptr<IAllocator> CreateCudaAllocator(int16_t device_id, size_t gpu_mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy, onnxruntime::CUDAExecutionProviderExternalAllocatorInfo& external_allocator_info, const OrtArenaCfg* default_memory_arena_cfg) override {
    return CUDAExecutionProvider::CreateCudaAllocator(device_id, gpu_mem_limit, arena_extend_strategy, external_allocator_info, default_memory_arena_cfg);
  }
} g_info;

struct CUDA_Provider : Provider {
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    // Calling a function like ::cudaDeviceSynchronize will cause CUDA to ensure there is binary code for the current GPU architecture
    // Ideally this will be already part of the binary, but if not, CUDA will JIT it during this call. This can take a very long time
    // (minutes even), so we want to detect when this happens and let the user know why so they can report it properly or even fix it.
    // See the linked issue in the warning message for more info
    {
      auto start_time = std::chrono::steady_clock::now();
      // Do a trivial cuda operation that will cause JIT to occur
      {
        void** cuda_memory{};
        ::cudaMalloc(&cuda_memory, 1);
        ::cudaFree(cuda_memory);
      }
      auto end_time = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
      if (duration > std::chrono::seconds{30}) {
        LOGS_DEFAULT(WARNING) << "CUDA took " << duration.count() << " seconds to start, please see this issue for how to fix it: https://github.com/microsoft/onnxruntime/issues/10746";
      }
    }

    auto params = reinterpret_cast<const OrtCUDAProviderOptionsV2*>(void_params);

    CUDAExecutionProviderInfo info{};
    info.device_id = gsl::narrow<OrtDevice::DeviceId>(params->device_id);
    info.gpu_mem_limit = params->gpu_mem_limit;
    info.arena_extend_strategy = params->arena_extend_strategy;
    info.cudnn_conv_algo_search = params->cudnn_conv_algo_search;
    info.do_copy_in_default_stream = params->do_copy_in_default_stream != 0;
    info.has_user_compute_stream = params->has_user_compute_stream != 0;
    info.user_compute_stream = params->user_compute_stream;
    info.default_memory_arena_cfg = params->default_memory_arena_cfg;
    info.cudnn_conv_use_max_workspace = params->cudnn_conv_use_max_workspace != 0;
    info.enable_cuda_graph = params->enable_cuda_graph != 0;
    info.cudnn_conv1d_pad_to_nc1d = params->cudnn_conv1d_pad_to_nc1d != 0;
    info.tunable_op.enable = params->tunable_op_enable;
    info.tunable_op.tuning_enable = params->tunable_op_tuning_enable;
    info.tunable_op.max_tuning_duration_ms = params->tunable_op_max_tuning_duration_ms;
    info.enable_skip_layer_norm_strict_mode = params->enable_skip_layer_norm_strict_mode != 0;

    return std::make_shared<CUDAProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto internal_options = onnxruntime::CUDAExecutionProviderInfo::FromProviderOptions(options);
    auto& cuda_options = *reinterpret_cast<OrtCUDAProviderOptionsV2*>(provider_options);

    cuda_options.device_id = internal_options.device_id;
    cuda_options.cudnn_conv_algo_search = internal_options.cudnn_conv_algo_search;
    cuda_options.gpu_mem_limit = internal_options.gpu_mem_limit;
    cuda_options.arena_extend_strategy = internal_options.arena_extend_strategy;
    cuda_options.do_copy_in_default_stream = internal_options.do_copy_in_default_stream;
    cuda_options.has_user_compute_stream = internal_options.has_user_compute_stream;
    cuda_options.user_compute_stream = internal_options.user_compute_stream;
    cuda_options.default_memory_arena_cfg = internal_options.default_memory_arena_cfg;
    cuda_options.cudnn_conv_use_max_workspace = internal_options.cudnn_conv_use_max_workspace;
    cuda_options.enable_cuda_graph = internal_options.enable_cuda_graph;
    cuda_options.cudnn_conv1d_pad_to_nc1d = internal_options.cudnn_conv1d_pad_to_nc1d;
    cuda_options.enable_skip_layer_norm_strict_mode = internal_options.enable_skip_layer_norm_strict_mode;
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtCUDAProviderOptionsV2*>(provider_options);
    return onnxruntime::CUDAExecutionProviderInfo::ToProviderOptions(options);
  }

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }

} g_provider;

CUDA_Provider* GetProvider() {
  return &g_provider;
}

}  // namespace onnxruntime
