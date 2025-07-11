// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_library_internal.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/framework/session_options.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ep_api.h"
#include "core/session/ort_apis.h"

#if defined(USE_DML)
#include "core/providers/dml/dml_provider_factory_creator.h"
#endif

#if defined(USE_WEBGPU)
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/session/allocator_adapters.h"
#endif

namespace onnxruntime {

class CpuEpFactory : public EpFactoryInternalImpl {
 public:
  CpuEpFactory() : EpFactoryInternalImpl(kCpuExecutionProvider, "Microsoft") {
  }

 private:
  OrtStatus* GetSupportedDevices(EpFactoryInternal& ep_factory,
                                 const OrtHardwareDevice* const* devices,
                                 size_t num_devices,
                                 OrtEpDevice** ep_devices,
                                 size_t max_ep_devices,
                                 size_t* p_num_ep_devices) noexcept override {
    size_t& num_ep_devices = *p_num_ep_devices;
    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];
      if (device.type == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
        ORT_API_RETURN_IF_ERROR(
            OrtExecutionProviderApi::CreateEpDevice(&ep_factory, &device, nullptr, nullptr,
                                                    &ep_devices[num_ep_devices++]));
      }
    }

    return nullptr;
  }

  OrtStatus* CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                      const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                      size_t num_devices,
                                      const OrtSessionOptions* session_options,
                                      const OrtLogger* session_logger,
                                      std::unique_ptr<IExecutionProvider>* ep) noexcept override {
    if (num_devices != 1) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "CPU EP factory currently only supports one device at a time.");
    }

    CPUExecutionProviderInfo epi{session_options->value.enable_cpu_mem_arena};
    *ep = std::make_unique<CPUExecutionProvider>(epi);
    (*ep)->SetLogger(session_logger->ToInternal());

    return nullptr;
  }
};

std::unique_ptr<EpLibraryInternal> EpLibraryInternal::CreateCpuEp() {
  auto cpu_factory_impl = std::make_unique<CpuEpFactory>();
  auto internal_factory = std::make_unique<EpFactoryInternal>(std::move(cpu_factory_impl));
  return std::make_unique<EpLibraryInternal>(std::move(internal_factory));
}

#if defined(USE_DML)
class DmlEpFactory : public EpFactoryInternalImpl {
 public:
  DmlEpFactory() : EpFactoryInternalImpl(kDmlExecutionProvider, "Microsoft") {
  }

 private:
  OrtStatus* GetSupportedDevices(EpFactoryInternal& ep_factory,
                                 const OrtHardwareDevice* const* devices,
                                 size_t num_devices,
                                 OrtEpDevice** ep_devices,
                                 size_t max_ep_devices,
                                 size_t* p_num_ep_devices) noexcept override {
    size_t& num_ep_devices = *p_num_ep_devices;
    num_ep_devices = 0;

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];
      if (device.type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
        std::unique_ptr<OrtKeyValuePairs> ep_options;

        // TODO: Should we ignore a user provided 'device_id' when they select an OrtEpDevice as that is
        //       associated with a specific device.
        //       How would we know what options should not allow user overrides if set in OrtEpDevice?
        int32_t device_id = 0;  // If no device_id was found default to 0
        if (auto it = device.metadata.Entries().find("DxgiAdapterNumber"); it != device.metadata.Entries().end()) {
          ep_options = std::make_unique<OrtKeyValuePairs>();
          device_id = std::stoi(it->second);
        }

        ep_options->Add("device_id", std::to_string(device_id));

        auto* api_status = OrtExecutionProviderApi::CreateEpDevice(&ep_factory,
                                                                   &device, nullptr, ep_options.get(),
                                                                   &ep_devices[num_ep_devices]);

        if (device_memory_infos.size() < device_id + 1) {
          device_memory_infos.resize(device_id + 1);
          device_allocators.resize(device_id + 1);
        }

        if (device_memory_infos[device_id] == nullptr) {
          // Create memory info for the device if it doesn't already exist
          device_memory_infos[device_id] = std::make_unique<OrtMemoryInfo>(
              "DML", OrtAllocatorType::OrtDeviceAllocator,
              OrtDevice(OrtDevice::DML, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::MICROSOFT,
                        narrow<int16_t>(device_id)));
        }

        // This is what we need to add once CreateAllocator is implemented to create a shared allocator for the device.
        // OrtExecutionProviderApi::EpDevice_AddAllocatorInfo(ep_devices[num_ep_devices],
        //                                                   device_memory_infos[device_id].get());

        if (api_status != nullptr) {
          return api_status;
        }

        ++num_ep_devices;
      }
    }

    return nullptr;
  }

  OrtStatus* CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                      const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                      size_t num_devices,
                                      const OrtSessionOptions* session_options,
                                      const OrtLogger* session_logger,
                                      std::unique_ptr<IExecutionProvider>* ep) noexcept override {
    *ep = nullptr;

    if (num_devices != 1) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "DML EP factory currently only supports one device at a time.");
    }

    auto ep_options = GetOptionsFromSessionOptions(session_options->value);
    auto dml_ep_factory = DMLProviderFactoryCreator::CreateFromProviderOptions(session_options->value.config_options,
                                                                               ep_options);

    *ep = dml_ep_factory->CreateProvider();
    (*ep)->SetLogger(session_logger->ToInternal());

    return nullptr;
  }

  OrtStatus* CreateAllocator(const OrtMemoryInfo* /*memory_info*/,
                             const OrtEp* /*ep*/,
                             const OrtKeyValuePairs* /*allocator_options*/,
                             OrtAllocator** allocator) noexcept override {
    // TODO: This needs to create an allocator for the specific device so it's available as a shared allocator. That
    //       requires pulling lots of things out of the DML EP to get the D3D12 device and create a
    //       BucketizedBufferAllocator. See providers\dml\DmlExecutionProvider\src\ExecutionProvider.cpp
    //*allocator = device_allocators[memory_info->device.Id()].get();
    *allocator = nullptr;
    return nullptr;
  }

  OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) override {
    // TODO: Wrap the IDataTransfer implementation so we can copy to device using OrtApi CopyTensors.
    *data_transfer = nullptr;
    return nullptr;
  }

  std::vector<std::unique_ptr<OrtMemoryInfo>> device_memory_infos;  // memory info for each device
  std::vector<std::unique_ptr<OrtAllocator>> device_allocators;     // allocators for each device
};

std::unique_ptr<EpLibraryInternal> EpLibraryInternal::CreateDmlEp() {
  auto dml_factory_impl = std::make_unique<DmlEpFactory>();
  auto internal_factory = std::make_unique<EpFactoryInternal>(std::move(dml_factory_impl));
  return std::make_unique<EpLibraryInternal>(std::move(internal_factory));
}
#endif

#if defined(USE_WEBGPU)
class WebGpuEpFactory : public EpFactoryInternalImpl {
 public:
  WebGpuEpFactory() : EpFactoryInternalImpl(kWebGpuExecutionProvider, "Microsoft"),
                      device_contexts_{nullptr, nullptr} {
  }

 private:
  // we create WebGpuContext instances that are only used for shared allocators but map to a specific device.
  // use device_id + this offset for the virtual device id in the WebGpuContext.
  static const int16_t kSharedAllocatorContextIdOffset = 16;

  static webgpu::WebGpuContext& CreateWebGpuContext(int16_t context_id) {
    webgpu::WebGpuContextConfig context_config{context_id,
                                               /*WGPUInstance*/ nullptr, /*WGPUDevice*/ nullptr,
                                               /*dawn_proc_table*/ nullptr,
                                               webgpu::ValidationMode::Disabled,
                                               /*preserve_device*/ false};

    webgpu::WebGpuBufferCacheConfig buffer_cache_config;
    buffer_cache_config.storage.mode = webgpu::BufferCacheMode::Disabled;
    buffer_cache_config.uniform.mode = webgpu::BufferCacheMode::Disabled;
    buffer_cache_config.query_resolve.mode = webgpu::BufferCacheMode::Disabled;
    buffer_cache_config.default_entry.mode = webgpu::BufferCacheMode::Disabled;

    int backend_type = 0;
#ifdef _WIN32
    // Setup Windows default backend type based on the build configuration
#if defined(DAWN_ENABLE_D3D12)
    backend_type = 4;  // WGPUBackendType_D3D12
#elif defined(DAWN_ENABLE_VULKAN)
    backend_type = 6;  // WGPUBackendType_Vulkan
#endif
#endif

    // ISSUE: CreateContext only creates one wgpu::Instance and that is behind a call_once.
    //        Any context id other than 0 is required to pass in the WGPUDevice and WGPUInstance
    //
    // We want to be able to create one high perf, and possibly one low power device.
    // We want to use offset context ids to keep the context for shared allocators separate from the context for
    // session allocators.

    auto& context = webgpu::WebGpuContextFactory::CreateContext(context_config);
    context.Initialize(buffer_cache_config, backend_type, /*enable_pix_capture*/ false);

    std::cout << "Initialized WebGPU context with backend type: " << backend_type << std::endl;

    return context;
  }

  OrtStatus* GetSupportedDevices(EpFactoryInternal& ep_factory,
                                 const OrtHardwareDevice* const* devices,
                                 size_t num_devices,
                                 OrtEpDevice** ep_devices,
                                 size_t max_ep_devices,
                                 size_t* p_num_ep_devices) noexcept override {
    size_t& num_ep_devices = *p_num_ep_devices;
    num_ep_devices = 0;

    // max we can have is 2 devices. one set to high perf, one set to low power
    // slot 0: discrete, slot 1:integrated, slot 2: first unknown seen
    std::vector<const OrtHardwareDevice*> gpu_devices{4, nullptr};

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];
      if (device.type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
        const auto& metadata = device.metadata.Entries();
        if (auto it = metadata.find("Discrete"); it != metadata.end()) {
          if (it->second == "1") {
            if (gpu_devices[0] == nullptr) {
              gpu_devices[0] = &device;
            }
          } else {
            if (gpu_devices[1] == nullptr) {
              gpu_devices[1] = &device;
            }
          }
        } else {
          // on linux we could maybe check if it's an nvidia card here and set to discrete if it is
          if (gpu_devices[2] == nullptr) {
            gpu_devices[2] = &device;
          }
        }
      }
    }

    // ignore the unknown if we have discrete and/or integrated.
    if (gpu_devices[2] && (gpu_devices[0] || gpu_devices[1])) {
      gpu_devices[2] = nullptr;  // we don't need the unknown one
    }

    // now the rules are simple. first device seen is HighPerformance, second is LowPower. max is 2 devices.
    int16_t device_id = 0;

    for (const OrtHardwareDevice* device : gpu_devices) {
      if (device == nullptr) {
        continue;
      }

      int16_t context_id = device_id + kSharedAllocatorContextIdOffset;
      auto& context = CreateWebGpuContext(context_id);
      device_contexts_[device_id] = &context;

      OrtKeyValuePairs ep_options;
      ep_options.Add("deviceId", std::to_string(device_id));
      ep_options.Add("powerPreference", device_id == 0 ? "HighPerformance" : "LowPower");

      OrtEpDevice* ep_device = ep_devices[num_ep_devices];
      ORT_API_RETURN_IF_ERROR(OrtExecutionProviderApi::CreateEpDevice(&ep_factory,
                                                                      device, nullptr, nullptr,
                                                                      &ep_device));

      // add memory info
      device_memory_info_[device_id] = OrtMemoryInfo(
          "WebGPU", OrtAllocatorType::OrtDeviceAllocator,
          OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, /*vendor id*/ 0, device_id));

      ep_device->device_memory_info = &device_memory_info_[device_id];

      ++num_ep_devices;
      ++device_id;
    }

    return nullptr;
  }

  OrtStatus* CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                      const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                      size_t num_devices,
                                      const OrtSessionOptions* session_options,
                                      const OrtLogger* session_logger,
                                      std::unique_ptr<IExecutionProvider>* ep) noexcept override {
    *ep = nullptr;

    if (num_devices != 1) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "WebGPU EP factory can only be called with one device.");
    }

    // need to read 'ep.webgpuexecutionprovider.deviceId' from session_options and only create a new factory
    // if device_factories_[device_id] is empty.
    // ISSUE: what if the session options are different? how is this handled currently?

    auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(session_options->value.config_options);
    *ep = webgpu_ep_factory->CreateProvider();
    (*ep)->SetLogger(session_logger->ToInternal());

    return nullptr;
  }

  OrtStatus* CreateAllocator(const OrtMemoryInfo* memory_info,
                             const OrtEp* /*ep*/,
                             const OrtKeyValuePairs* /*allocator_options*/,
                             OrtAllocator** allocator) noexcept override {
    const int16_t device_id = memory_info->device.Id();

    // we expect an address match
    if (memory_info != &device_memory_info_[device_id]) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "Unexpected OrtMemoryInfo value. "
                                   "Does not match a value added to an OrtEpDevice returned by this factory");
    }

    std::unique_ptr<OrtAllocator>& device_allocator = device_allocators_[device_id];

    if (device_allocator == nullptr) {
      const auto& buffer_manager = device_contexts_[device_id]->BufferManager();
      auto webgpu_allocator = std::make_unique<webgpu::GpuBufferAllocator>(buffer_manager);
      device_allocator = std::make_unique<OrtAllocatorImplWrappingIAllocator>(std::move(webgpu_allocator));
    }

    *allocator = device_allocator.get();
    return nullptr;
  }

  /*
  OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) override {
    // TODO: Wrap the IDataTransfer implementation so we can copy to device using OrtApi CopyTensors.
    *data_transfer = nullptr;
    return nullptr;
  }
  */
 private:
  // WebGpuContext that we use for shared allocators
  std::array<webgpu::WebGpuContext*, 2> device_contexts_;
  std::array<OrtMemoryInfo, 2> device_memory_info_;
  std::array<std::unique_ptr<OrtAllocator>, 2> device_allocators_;
  std::array<std::shared_ptr<IExecutionProviderFactory>, 2> device_factories_;
};

std::unique_ptr<EpLibraryInternal> EpLibraryInternal::CreateWebGpuEp() {
  auto webgpu_factory_impl = std::make_unique<WebGpuEpFactory>();
  auto internal_factory = std::make_unique<EpFactoryInternal>(std::move(webgpu_factory_impl));
  return std::make_unique<EpLibraryInternal>(std::move(internal_factory));
}
#endif

std::vector<std::unique_ptr<EpLibraryInternal>> EpLibraryInternal::CreateInternalEps() {
  std::vector<std::unique_ptr<EpLibraryInternal>> internal_eps;
  internal_eps.reserve(4);

  // CPU EP
  internal_eps.push_back(CreateCpuEp());

#if defined(USE_WEBGPU)
  internal_eps.push_back(CreateWebGpuEp());
#endif

#if defined(USE_DML)
  internal_eps.push_back(CreateDmlEp());
#endif

  return internal_eps;
}

}  // namespace onnxruntime
