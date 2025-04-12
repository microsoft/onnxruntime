// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
#include <atomic>
#include <utility>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/amdgpu/amdgpu_provider_factory.h"
#include "core/providers/amdgpu/amdgpu_execution_provider.h"
#include "core/providers/amdgpu/amdgpu_execution_provider_info.h"
#include "core/providers/amdgpu/amdgpu_provider_factory_creator.h"
#include "core/providers/amdgpu/amdgpu_allocator.h"
#include "core/providers/amdgpu/amdgpu_data_transfer.h"
#include "core/framework/provider_options.h"

#include "core/session/onnxruntime_c_api.h"
#include "core/providers/amdgpu/amdgpu_call.h"

using namespace onnxruntime;

namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct AMDGPUProviderFactory final : IExecutionProviderFactory {
  explicit AMDGPUProviderFactory(AMDGPUExecutionProviderInfo info) : info_{std::move(info)} {}
  ~AMDGPUProviderFactory() override = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  AMDGPUExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> AMDGPUProviderFactory::CreateProvider() {
  return std::make_unique<AMDGPUExecutionProvider>(info_);
}

struct ProviderInfo_AMDGPU_Impl final : ProviderInfo_AMDGPU {
  std::unique_ptr<IAllocator> CreateAMDGPUAllocator(OrtDevice::DeviceId device_id, const char* name) override {
    return std::make_unique<AMDGPUAllocator>(device_id, name);
  }

  std::unique_ptr<IAllocator> CreateAMDGPUPinnedAllocator(OrtDevice::DeviceId device_id, const char* name) override {
    return std::make_unique<AMDGPUPinnedAllocator>(device_id, name);
  }

  void AMDGPUMemcpy_HostToDevice(void* dst, const void* src, size_t count) override {
    // hipMemcpy() operates on the default stream
    HIP_CALL_THROW(hipMemcpy(dst, src, count, hipMemcpyHostToDevice));

    // To ensure that the copy has completed, invoke a stream sync for the default stream.
    // For transfers from pageable host memory to device memory, a stream sync is performed before the copy is initiated.
    // The function will return once the pageable buffer has been copied to the staging memory for DMA transfer
    // to device memory, but the DMA to final destination may not have completed.

    HIP_CALL_THROW(hipStreamSynchronize(0));
  }

  // Used by onnxruntime_pybind_state.cc
  void AMDGPUMemcpy_DeviceToHost(void* dst, const void* src, size_t count) override {
    // For transfers from device to either pageable or pinned host memory, the function returns only once the copy has completed.
    HIP_CALL_THROW(hipMemcpy(dst, src, count, hipMemcpyDeviceToHost));
  }

  std::shared_ptr<IAllocator> CreateAMDGPUAllocator(OrtDevice::DeviceId device_id, const size_t migx_mem_limit, ArenaExtendStrategy arena_extend_strategy, AMDGPUExecutionProviderExternalAllocatorInfo& external_allocator_info, const OrtArenaCfg* default_memory_arena_cfg) override {
    return AMDGPUExecutionProvider::CreateAMDGPUAllocator(device_id, migx_mem_limit, arena_extend_strategy, external_allocator_info, default_memory_arena_cfg);
  }
} g_info;

struct AMDGPU_Provider final : Provider {
  virtual ~AMDGPU_Provider() = default;
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    AMDGPUExecutionProviderInfo info;
    info.device_id = static_cast<OrtDevice::DeviceId>(device_id);
    info.target_device = "gpu";
    return std::make_shared<AMDGPUProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    auto& options = *static_cast<const OrtAMDGPUProviderOptions*>(provider_options);
    AMDGPUExecutionProviderInfo info;
    info.device_id = static_cast<OrtDevice::DeviceId>(options.device_id);
    info.target_device = "gpu";
    info.fp16_enable = options.amdgpu_fp16_enable;
    info.fp8_enable = options.amdgpu_fp8_enable;
    info.exhaustive_tune = options.amdgpu_exhaustive_tune;
    info.int8_enable = options.amdgpu_int8_enable;
    info.int8_calibration_table_name = "";
    if (options.amdgpu_int8_calibration_table_name != nullptr) {
      info.int8_calibration_table_name = options.amdgpu_int8_calibration_table_name;
    }
    info.int8_use_native_calibration_table = options.amdgpu_use_native_calibration_table != 0;
    info.model_cache_dir = "";
    if (options.amdgpu_cache_dir != nullptr) {
      info.model_cache_dir = options.amdgpu_cache_dir;
    }
    info.arena_extend_strategy = static_cast<onnxruntime::ArenaExtendStrategy>(options.amdgpu_arena_extend_strategy);
    info.mem_limit = options.amdgpu_mem_limit;
    return std::make_shared<AMDGPUProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto internal_options = AMDGPUExecutionProviderInfo::FromProviderOptions(options);
    auto& migx_options = *static_cast<OrtAMDGPUProviderOptions*>(provider_options);
    migx_options.device_id = internal_options.device_id;
    migx_options.amdgpu_fp16_enable = internal_options.fp16_enable;
    migx_options.amdgpu_fp8_enable = internal_options.fp8_enable;
    migx_options.amdgpu_int8_enable = internal_options.int8_enable;
    migx_options.amdgpu_exhaustive_tune = internal_options.exhaustive_tune;

    char* dest = nullptr;
    auto str_size = internal_options.int8_calibration_table_name.size();
    if (str_size == 0) {
      migx_options.amdgpu_int8_calibration_table_name = nullptr;
    } else {
      dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.int8_calibration_table_name.c_str(), str_size);
#else
      strncpy(dest, internal_options.int8_calibration_table_name.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      migx_options.amdgpu_int8_calibration_table_name = dest;
    }

    migx_options.amdgpu_use_native_calibration_table = internal_options.int8_use_native_calibration_table;
    migx_options.amdgpu_cache_dir = internal_options.model_cache_dir.string().c_str();
    migx_options.amdgpu_arena_extend_strategy = static_cast<int>(internal_options.arena_extend_strategy);
    migx_options.amdgpu_mem_limit = internal_options.mem_limit;
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *static_cast<const OrtAMDGPUProviderOptions*>(provider_options);
    return AMDGPUExecutionProviderInfo::ToProviderOptions(options);
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
  return &g_provider;
}
}
