// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <shlwapi.h>
#endif

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/migraphx_provider_factory.h"
#include "core/providers/migraphx/migraphx_execution_provider.h"
#include "core/providers/migraphx/migraphx_execution_provider_info.h"
#include "core/providers/migraphx/migraphx_allocator.h"
#include "core/providers/migraphx/gpu_data_transfer.h"
#include "core/framework/provider_options.h"

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct MIGraphXProviderFactory : IExecutionProviderFactory {
  explicit MIGraphXProviderFactory(MIGraphXExecutionProviderInfo info) : info_{std::move(info)} {}
  ~MIGraphXProviderFactory() override = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  MIGraphXExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> MIGraphXProviderFactory::CreateProvider() {
  return std::make_unique<MIGraphXExecutionProvider>(info_);
}

struct ProviderInfo_MIGraphX_Impl final : ProviderInfo_MIGraphX {
  std::unique_ptr<IAllocator> CreateMIGraphXAllocator(OrtDevice::DeviceId device_id, const char* name) override {
    return std::make_unique<MIGraphXAllocator>(device_id, name);
  }

  std::unique_ptr<IAllocator> CreateMIGraphXPinnedAllocator(OrtDevice::DeviceId device_id, const char* name) override {
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

  std::shared_ptr<IAllocator> CreateMIGraphXAllocator(OrtDevice::DeviceId device_id, size_t mem_limit, onnxruntime::ArenaExtendStrategy arena_extend_strategy,
                                                      void* alloc_fn, void* free_fn, void* empty_cache_fn, const OrtArenaCfg* default_memory_arena_cfg) override {
    if (alloc_fn != nullptr && free_fn != nullptr) {
      AllocatorCreationInfo default_memory_info{
          [alloc_fn, free_fn, empty_cache_fn](OrtDevice::DeviceId id) {
            return std::make_unique<MIGraphXExternalAllocator>(id, HIP, alloc_fn, free_fn, empty_cache_fn);
          },
          device_id, false};

      return CreateAllocator(default_memory_info);
    }
    AllocatorCreationInfo default_memory_info{
        [](OrtDevice::DeviceId id) {
          return std::make_unique<MIGraphXAllocator>(id, HIP);
        },
        device_id,
        true,
        {default_memory_arena_cfg ? *default_memory_arena_cfg
                                  : OrtArenaCfg(mem_limit, static_cast<int>(arena_extend_strategy),
                                                -1, -1, -1, -1L)},
        // make it stream aware
        true};

    // ROCM malloc/free is expensive so always use an arena
    return CreateAllocator(default_memory_info);
  }
} g_info;

struct MIGraphX_Provider final : Provider {
  void* GetInfo() override { return &g_info; }

  virtual ~MIGraphX_Provider() = default;

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    MIGraphXExecutionProviderInfo info;
    info.device_id = static_cast<OrtDevice::DeviceId>(device_id);
    info.target_device = "gpu";
    return std::make_shared<MIGraphXProviderFactory>(info);
  }

  // Method uses ProviderOptions, and not OrtMIGraphXProviderOptions (obsolete)
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    if (provider_options != nullptr) {
      return std::make_shared<MIGraphXProviderFactory>(
          MIGraphXExecutionProviderInfo{*static_cast<const ProviderOptions*>(provider_options)});
    }
    return nullptr;
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    MIGraphXExecutionProviderInfo internal_options{options};
    const auto migx_options = static_cast<OrtMIGraphXProviderOptions*>(provider_options);
    migx_options->device_id = internal_options.device_id;
    migx_options->migraphx_fp16_enable = internal_options.fp16_enable;
    migx_options->migraphx_fp8_enable = internal_options.fp8_enable;
    migx_options->migraphx_int8_enable = internal_options.int8_enable;
    migx_options->migraphx_exhaustive_tune = internal_options.exhaustive_tune;

    if (internal_options.int8_calibration_table_name.empty()) {
      migx_options->migraphx_int8_calibration_table_name = nullptr;
    } else {
      auto str_size = internal_options.int8_calibration_table_name.size();
      auto dest = new char[str_size + 1];
#ifdef _MSC_VER
      strncpy_s(dest, str_size + 1, internal_options.int8_calibration_table_name.c_str(), str_size);
#else
      strncpy(dest, internal_options.int8_calibration_table_name.c_str(), str_size);
#endif
      dest[str_size] = '\0';
      migx_options->migraphx_int8_calibration_table_name = static_cast<const char*>(dest);
    }

    migx_options->migraphx_use_native_calibration_table = internal_options.int8_use_native_calibration_table;

    migx_options->migraphx_arena_extend_strategy = static_cast<int>(internal_options.arena_extend_strategy);
    migx_options->migraphx_mem_limit = internal_options.mem_limit;
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    return provider_options != nullptr ? MIGraphXExecutionProviderInfo{
                                             *static_cast<const OrtMIGraphXProviderOptions*>(provider_options)}
                                             .ToProviderOptions()
                                       : ProviderOptions{};
  }

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                  const OrtKeyValuePairs* const* /*ep_metadata*/,
                                  size_t num_devices,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    ORT_UNUSED_PARAMETER(num_devices);
    const auto ep_factory = CreateExecutionProviderFactory(&provider_options);
    ep = ep_factory->CreateProvider(session_options, logger);
    return Status::OK();
  }

  void Initialize() override {
#ifdef _WIN32
    HMODULE module = nullptr;
    if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          static_cast<LPCSTR>(static_cast<void*>(InitializeRegistry)),
                          &module) != 0) {
      std::vector<wchar_t> pathBuf;
      for (;;) {
        pathBuf.resize(pathBuf.size() + MAX_PATH);
        if (const auto writen = GetModuleFileNameW(module, pathBuf.data(), static_cast<DWORD>(pathBuf.size())); writen < pathBuf.size()) {
          break;
        }
      }
      std::filesystem::path path(pathBuf.begin(), pathBuf.end());
      SetDllDirectoryW(path.parent_path().native().c_str());
    }
#endif
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
    ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;

    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;

    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;
    CreateDataTransfer = CreateDataTransferImpl;

    IsStreamAware = IsStreamAwareImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
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

  static uint32_t GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const MigraphXEpFactory*>(this_ptr);
    return factory->vendor_id;
  }

  static const char* GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const MigraphXEpFactory*>(this_ptr);
    return factory->version.c_str();
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

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* /*memory_info*/,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept {
    auto* factory = static_cast<MigraphXEpFactory*>(this_ptr);

    *allocator = nullptr;
    return factory->ort_api.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "CreateAllocator should not be called as we did not add OrtMemoryInfo to our OrtEpDevice.");
  }

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/, OrtAllocator* /*allocator*/) noexcept {
    // should never be called as we don't implement CreateAllocator
  }

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* /*this_ptr*/,
                                                        OrtDataTransferImpl** data_transfer) noexcept {
    *data_transfer = nullptr;  // not implemented
    return nullptr;
  }

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return false;
  }

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                               const OrtMemoryDevice* /*memory_device*/,
                                                               const OrtKeyValuePairs* /*stream_options*/,
                                                               OrtSyncStreamImpl** stream) noexcept {
    auto* factory = static_cast<MigraphXEpFactory*>(this_ptr);

    *stream = nullptr;
    return factory->ort_api.CreateStatus(
        ORT_INVALID_ARGUMENT, "CreateSyncStreamForDevice should not be called as IsStreamAware returned false.");
  }

  const OrtApi& ort_api;
  const OrtLogger& default_logger;
  const std::string ep_name;
  const std::string vendor{"AMD"};
  const std::string version{"1.0.0"};  // MigraphX EP version

  // Not using AMD vendor id 0x1002 so that OrderDevices in provider_policy_context.cc will default dml ep
  const uint32_t vendor_id{0x9999};
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
