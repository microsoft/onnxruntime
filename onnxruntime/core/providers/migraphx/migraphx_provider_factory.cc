// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
#include <atomic>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/migraphx_provider_factory.h"
#include "migraphx_execution_provider.h"
#include "migraphx_execution_provider_info.h"
#include "migraphx_provider_factory_creator.h"
#include "migraphx_allocator.h"
#include "gpu_data_transfer.h"
#include "core/framework/provider_options.h"

#include "core/session/onnxruntime_c_api.h"

using namespace onnxruntime;

namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct MIGraphXProviderFactory : IExecutionProviderFactory {
  MIGraphXProviderFactory(const MIGraphXExecutionProviderInfo& info) : info_{info} {}
  ~MIGraphXProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  MIGraphXExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> MIGraphXProviderFactory::CreateProvider() {
  return std::make_unique<MIGraphXExecutionProvider>(info_);
}

struct ProviderInfo_MIGraphX_Impl final : ProviderInfo_MIGraphX {
  std::unique_ptr<IAllocator> CreateMIGraphXAllocator(int16_t device_id, const char* name) override {
    return std::make_unique<MIGraphXAllocator>(device_id, name);
  }

  std::unique_ptr<IAllocator> CreateMIGraphXPinnedAllocator(int16_t device_id, const char* name) override {
    return std::make_unique<MIGraphXPinnedAllocator>(device_id, name);
  }

  void MIGraphXMemcpy_HostToDevice(void* dst, const void* src, size_t count) override {
    // hipMemcpy() operates on the default stream
    HIP_CALL_THROW(hipMemcpy(dst, src, count, hipMemcpyHostToDevice));

    // To ensure that the copy has completed, invoke a stream sync for the default stream.
    // For transfers from pageable host memory to device memory, a stream sync is performed before the copy is initiated.
    // The function will return once the pageable buffer has been copied to the staging memory for DMA transfer
    // to device memory, but the DMA to final destination may not have completed.

    HIP_CALL_THROW(hipStreamSynchronize(0));
  }

  // Used by onnxruntime_pybind_state.cc
  void MIGraphXMemcpy_DeviceToHost(void* dst, const void* src, size_t count) override {
    // For transfers from device to either pageable or pinned host memory, the function returns only once the copy has completed.
    HIP_CALL_THROW(hipMemcpy(dst, src, count, hipMemcpyDeviceToHost));
  }

  std::shared_ptr<IAllocator> CreateMIGraphXAllocator(int16_t device_id, size_t migx_mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy, onnxruntime::MIGraphXExecutionProviderExternalAllocatorInfo& external_allocator_info, const OrtArenaCfg* default_memory_arena_cfg) override {
    return MIGraphXExecutionProvider::CreateMIGraphXAllocator(device_id, migx_mem_limit, arena_extend_strategy, external_allocator_info, default_memory_arena_cfg);
  }
} g_info;

struct MIGraphX_Provider : Provider {
  void* GetInfo() override { return &g_info; }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    MIGraphXExecutionProviderInfo info;
    info.device_id = static_cast<OrtDevice::DeviceId>(device_id);
    info.target_device = "gpu";
    return std::make_shared<MIGraphXProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtMIGraphXProviderOptions*>(provider_options);
    MIGraphXExecutionProviderInfo info;
    info.device_id = static_cast<OrtDevice::DeviceId>(options.device_id);
    info.target_device = "gpu";
    info.fp16_enable = options.migraphx_fp16_enable;
    info.fp8_enable = options.migraphx_fp8_enable;
    info.exhaustive_tune = options.migraphx_exhaustive_tune;
    info.int8_enable = options.migraphx_int8_enable;
    info.int8_calibration_table_name = "";
    if (options.migraphx_int8_calibration_table_name != nullptr) {
      info.int8_calibration_table_name = options.migraphx_int8_calibration_table_name;
    }
    info.int8_use_native_calibration_table = options.migraphx_use_native_calibration_table != 0;
    info.save_compiled_model = options.migraphx_save_compiled_model;
    info.save_model_file = "";
    if (options.migraphx_save_model_path != nullptr) {
      info.save_model_file = options.migraphx_save_model_path;
    }
    info.load_compiled_model = options.migraphx_load_compiled_model;
    info.load_model_file = "";
    if (options.migraphx_load_model_path != nullptr) {
      info.load_model_file = options.migraphx_load_model_path;
    }
    info.arena_extend_strategy = static_cast<onnxruntime::ArenaExtendStrategy>(options.migraphx_arena_extend_strategy);
    info.mem_limit = options.migraphx_mem_limit;
    return std::make_shared<MIGraphXProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    auto internal_options = onnxruntime::MIGraphXExecutionProviderInfo::FromProviderOptions(options);
    auto& migx_options = *reinterpret_cast<OrtMIGraphXProviderOptions*>(provider_options);
    migx_options.device_id = internal_options.device_id;
    migx_options.migraphx_fp16_enable = internal_options.fp16_enable;
    migx_options.migraphx_fp8_enable = internal_options.fp8_enable;
    migx_options.migraphx_int8_enable = internal_options.int8_enable;
    migx_options.migraphx_exhaustive_tune = internal_options.exhaustive_tune;

    char* dest = nullptr;
    auto str_size = internal_options.int8_calibration_table_name.size();
    if (str_size == 0) {
      migx_options.migraphx_int8_calibration_table_name = nullptr;
    } else {
      dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.int8_calibration_table_name.c_str(), str_size);
#else
      strncpy(dest, internal_options.int8_calibration_table_name.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      migx_options.migraphx_int8_calibration_table_name = (const char*)dest;
    }

    migx_options.migraphx_use_native_calibration_table = internal_options.int8_use_native_calibration_table;

    migx_options.migraphx_save_compiled_model = internal_options.save_compiled_model;
    migx_options.migraphx_save_model_path = internal_options.save_model_file.c_str();
    migx_options.migraphx_load_compiled_model = internal_options.load_compiled_model;
    migx_options.migraphx_load_model_path = internal_options.load_model_file.c_str();
    migx_options.migraphx_arena_extend_strategy = static_cast<int>(internal_options.arena_extend_strategy);
    migx_options.migraphx_mem_limit = internal_options.mem_limit;
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtMIGraphXProviderOptions*>(provider_options);
    return onnxruntime::MIGraphXExecutionProviderInfo::ToProviderOptions(options);
  }

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                  const OrtKeyValuePairs* const* /*ep_metadata*/,
                                  size_t num_devices,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    ORT_UNUSED_PARAMETER(num_devices);
    const ConfigOptions* config_options = &session_options.GetConfigOptions();

    std::array<const void*, 2> configs_array = {&provider_options, config_options};
    auto ep_factory = CreateExecutionProviderFactory(&provider_options);
    ep = ep_factory->CreateProvider(session_options, logger);

    return Status::OK();
  }

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }

} g_provider;

}  // namespace onnxruntime

#include "core/framework/error_code_helper.h"

// OrtEpApi infrastructure to be able to use the MigraphX/AMDGPU EP as an OrtEpFactory for auto EP selection.
struct MigraphXEpFactory : OrtEpFactory {
  MigraphXEpFactory(const OrtApi& ort_api_in,
                    const char* ep_name,
                    OrtHardwareDeviceType hw_type,
                    const OrtLogger& default_logger_in)
      : ort_api{ort_api_in}, default_logger{default_logger_in}, ep_name{ep_name}, ort_hw_device_type{hw_type} {
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVendorId = GetVendorIdImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }

  // Returns the name for the EP. Each unique factory configuration must have a unique name.
  // Ex: a factory that supports NPU should have a different than a factory that supports GPU.
  static const char* GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const MigraphXEpFactory*>(this_ptr);
    return factory->ep_name.c_str();
  }

  static const char* GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const MigraphXEpFactory*>(this_ptr);
    return factory->vendor.c_str();
  }

  static uint32_t GetVendorIdImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const MigraphXEpFactory*>(this_ptr);
    return factory->vendor_id;
  }

  // Creates and returns OrtEpDevice instances for all OrtHardwareDevices that this factory supports.
  // An EP created with this factory is expected to be able to execute a model with *all* supported
  // hardware devices at once. A single instance of MigraphX EP is not currently setup to partition a model among
  // multiple different MigraphX backends at once (e.g, npu, cpu, gpu), so this factory instance is set to only
  // support one backend: gpu. To support a different backend, like npu, create a different factory instance
  // that only supports NPU.
  static OrtStatus* GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* p_num_ep_devices) noexcept {
    size_t& num_ep_devices = *p_num_ep_devices;
    auto* factory = static_cast<MigraphXEpFactory*>(this_ptr);

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];
      if (factory->ort_api.HardwareDevice_Type(&device) == factory->ort_hw_device_type) {
        // factory->ort_api.HardwareDevice_VendorId(&device) == factory->vendor_id) {
        OrtKeyValuePairs* ep_options = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_options);
        ORT_API_RETURN_IF_ERROR(
            factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, nullptr, ep_options,
                                                        &ep_devices[num_ep_devices++]));
      }
    }

    return nullptr;
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* /*this_ptr*/,
                                 _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                 _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                 _In_ size_t /*num_devices*/,
                                 _In_ const OrtSessionOptions* /*session_options*/,
                                 _In_ const OrtLogger* /*logger*/,
                                 _Out_ OrtEp** /*ep*/) noexcept {
    return onnxruntime::CreateStatus(ORT_INVALID_ARGUMENT, "[MigraphX/AMDGPU EP] EP factory does not support this method.");
  }

  static void ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* /*ep*/) noexcept {
    // no-op as we never create an EP here.
  }

  const OrtApi& ort_api;
  const OrtLogger& default_logger;
  const std::string ep_name;
  const std::string vendor{"AMD"};

  const uint32_t vendor_id{0x1002};
  const OrtHardwareDeviceType ort_hw_device_type;  // Supported OrtHardwareDevice
};

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);

  // Factory could use registration_name or define its own EP name.
  auto factory_gpu = std::make_unique<MigraphXEpFactory>(*ort_api,
                                                         onnxruntime::kMIGraphXExecutionProvider,
                                                         OrtHardwareDeviceType_GPU,
                                                         *default_logger);

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory_gpu.release();
  *num_factories = 1;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<MigraphXEpFactory*>(factory);
  return nullptr;
}

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
