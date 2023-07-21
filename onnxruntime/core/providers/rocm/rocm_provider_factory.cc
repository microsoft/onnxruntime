// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_provider_factory.h"

#include <memory>

#include "core/common/gsl.h"

#include "core/providers/rocm/rocm_execution_provider.h"
#include "core/providers/rocm/rocm_execution_provider_info.h"
#include "core/providers/rocm/rocm_allocator.h"
#include "core/providers/rocm/rocm_provider_factory_creator.h"
#include "core/providers/rocm/gpu_data_transfer.h"
#include "core/providers/rocm/math/unary_elementwise_ops_impl.h"

#if defined(USE_ROCM) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
#include "orttraining/training_ops/rocm/communication/nccl_service.h"
#endif

using namespace onnxruntime;

namespace onnxruntime {

#if defined(USE_ROCM) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
namespace rocm {
rocm::INcclService& GetINcclService();
}
#endif

void InitializeRegistry();
void DeleteRegistry();

struct ROCMProviderFactory : IExecutionProviderFactory {
  ROCMProviderFactory(const ROCMExecutionProviderInfo& info)
      : info_{info} {}
  ~ROCMProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  ROCMExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> ROCMProviderFactory::CreateProvider() {
  return std::make_unique<ROCMExecutionProvider>(info_);
}

struct ProviderInfo_ROCM_Impl : ProviderInfo_ROCM {
  OrtStatus* SetCurrentGpuDeviceId(_In_ int device_id) override {
    int num_devices;
    auto hip_err = ::hipGetDeviceCount(&num_devices);
    if (hip_err != hipSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to set device id since hipGetDeviceCount failed.");
    }

    if (device_id >= num_devices) {
      std::ostringstream ostr;
      ostr << "Invalid device id. Device id should be less than total number of devices (" << num_devices << ")";
      return CreateStatus(ORT_INVALID_ARGUMENT, ostr.str().c_str());
    }

    hip_err = hipSetDevice(device_id);
    if (hip_err != hipSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to set device id.");
    }
    return nullptr;
  }

  OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) override {
    auto hip_err = hipGetDevice(device_id);
    if (hip_err != hipSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to get device id.");
    }
    return nullptr;
  }

  std::unique_ptr<IAllocator> CreateROCMAllocator(int16_t device_id, const char* name) override {
    return std::make_unique<ROCMAllocator>(device_id, name);
  }

  std::unique_ptr<IAllocator> CreateROCMPinnedAllocator(const char* name) override {
    return std::make_unique<ROCMPinnedAllocator>(name);
  }

  std::unique_ptr<IDataTransfer> CreateGPUDataTransfer() override {
    return std::make_unique<GPUDataTransfer>();
  }

  void rocm__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) override {
    return rocm::Impl_Cast(static_cast<hipStream_t>(stream), input_data, output_data, count);
  }

  void rocm__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) override {
    return rocm::Impl_Cast(static_cast<hipStream_t>(stream), input_data, output_data, count);
  }

  void rocm__Impl_Cast(void* stream, const double* input_data, float* output_data, size_t count) override {
    return rocm::Impl_Cast(static_cast<hipStream_t>(stream), input_data, output_data, count);
  }

  void rocm__Impl_Cast(void* stream, const float* input_data, double* output_data, size_t count) override {
    return rocm::Impl_Cast(static_cast<hipStream_t>(stream), input_data, output_data, count);
  }

  Status RocmCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { return RocmCall<hipError_t, false>(hipError_t(retCode), exprString, libName, hipError_t(successCode), msg, file, line); }
  void RocmCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { RocmCall<hipError_t, true>(hipError_t(retCode), exprString, libName, hipError_t(successCode), msg, file, line); }

  void CopyGpuToCpu(void* dst_ptr, const void* src_ptr, const size_t size, const OrtMemoryInfo& dst_location, const OrtMemoryInfo& src_location) override {
    ORT_ENFORCE(dst_location.device.Type() == OrtDevice::CPU);

    // Current ROCM device.
    int device;
    HIP_CALL_THROW(hipGetDevice(&device));

    if (device != src_location.id) {
      // Need to switch to the allocating device.
      HIP_CALL_THROW(hipSetDevice(src_location.id));
      // Copy from GPU to CPU.
      HIP_CALL_THROW(hipMemcpy(dst_ptr, src_ptr, size, hipMemcpyDeviceToHost));
      // Switch back to current device.
      HIP_CALL_THROW(hipSetDevice(device));
    } else {
      // Copy from GPU to CPU.
      HIP_CALL_THROW(hipMemcpy(dst_ptr, src_ptr, size, hipMemcpyDeviceToHost));
    }
  }

  // Used by slice_concatenate_test.cc and onnxruntime_pybind_state.cc
  void rocmMemcpy_HostToDevice(void* dst, const void* src, size_t count) override { HIP_CALL_THROW(hipMemcpy(dst, src, count, hipMemcpyHostToDevice)); }
  // Used by onnxruntime_pybind_state.cc
  void rocmMemcpy_DeviceToHost(void* dst, const void* src, size_t count) override { HIP_CALL_THROW(hipMemcpy(dst, src, count, hipMemcpyDeviceToHost)); }

  int hipGetDeviceCount() override {
    int num_devices = 0;
    HIP_CALL_THROW(::hipGetDeviceCount(&num_devices));
    return num_devices;
  }

  void ROCMExecutionProviderInfo__FromProviderOptions(const ProviderOptions& options, ROCMExecutionProviderInfo& info) override {
    info = ROCMExecutionProviderInfo::FromProviderOptions(options);
  }

#if defined(USE_ROCM) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
  rocm::INcclService& GetINcclService() override {
    return rocm::GetINcclService();
  }
#endif

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const ROCMExecutionProviderInfo& info) override {
    return std::make_shared<ROCMProviderFactory>(info);
  }

  std::shared_ptr<IAllocator> CreateRocmAllocator(int16_t device_id, size_t gpu_mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy, onnxruntime::ROCMExecutionProviderExternalAllocatorInfo& external_allocator_info, OrtArenaCfg* default_memory_arena_cfg) override {
    return ROCMExecutionProvider::CreateRocmAllocator(device_id, gpu_mem_limit, arena_extend_strategy, external_allocator_info, default_memory_arena_cfg);
  }

} g_info;

struct ROCM_Provider : Provider {
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* void_params) override {
    auto params = reinterpret_cast<const OrtROCMProviderOptions*>(void_params);

    ROCMExecutionProviderInfo info{};
    info.device_id = gsl::narrow<OrtDevice::DeviceId>(params->device_id);
    info.gpu_mem_limit = params->gpu_mem_limit;
    info.arena_extend_strategy = static_cast<onnxruntime::ArenaExtendStrategy>(params->arena_extend_strategy);
    info.miopen_conv_exhaustive_search = params->miopen_conv_exhaustive_search;
    info.do_copy_in_default_stream = params->do_copy_in_default_stream;
    info.has_user_compute_stream = params->has_user_compute_stream;
    info.user_compute_stream = params->user_compute_stream;
    info.default_memory_arena_cfg = params->default_memory_arena_cfg;
    info.tunable_op.enable = params->tunable_op_enable;
    info.tunable_op.tuning_enable = params->tunable_op_tuning_enable;
    info.tunable_op.max_tuning_duration_ms = params->tunable_op_max_tuning_duration_ms;

    return std::make_shared<ROCMProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto info = onnxruntime::ROCMExecutionProviderInfo::FromProviderOptions(options);
    auto& rocm_options = *reinterpret_cast<OrtROCMProviderOptions*>(provider_options);

    rocm_options.device_id = info.device_id;
    rocm_options.gpu_mem_limit = info.gpu_mem_limit;
    rocm_options.arena_extend_strategy = static_cast<int>(info.arena_extend_strategy);
    rocm_options.miopen_conv_exhaustive_search = info.miopen_conv_exhaustive_search;
    rocm_options.do_copy_in_default_stream = info.do_copy_in_default_stream;
    rocm_options.has_user_compute_stream = info.has_user_compute_stream;
    rocm_options.user_compute_stream = info.user_compute_stream;
    rocm_options.default_memory_arena_cfg = info.default_memory_arena_cfg;
    rocm_options.tunable_op_enable = info.tunable_op.enable;
    rocm_options.tunable_op_tuning_enable = info.tunable_op.tuning_enable;
    rocm_options.tunable_op_max_tuning_duration_ms = info.tunable_op.max_tuning_duration_ms;
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtROCMProviderOptions*>(provider_options);
    return onnxruntime::ROCMExecutionProviderInfo::ToProviderOptions(options);
  }

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
