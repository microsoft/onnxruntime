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
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
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

        if (device_memory_infos.size() < device_id) {
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
  WebGpuEpFactory() : EpFactoryInternalImpl(kWebGpuExecutionProvider, "Microsoft") {
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
        // TODO: any metadata or options to add?
        ORT_API_RETURN_IF_ERROR(OrtExecutionProviderApi::CreateEpDevice(&ep_factory,
                                                                        &device, nullptr, nullptr,
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
    *ep = nullptr;

    if (num_devices != 1) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "WebGPU EP factory currently only supports one device at a time.");
    }

    auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(session_options->value.config_options);
    *ep = webgpu_ep_factory->CreateProvider();
    (*ep)->SetLogger(session_logger->ToInternal());

    return nullptr;
  }

  /* TODO: Implement CreateAllocator and CreateDataTransfer to support shared allocators and data transfer outside of
           an InferenceSession.
  OrtStatus* CreateAllocator(const OrtMemoryInfo* memory_info,
                             const OrtKeyValuePairs* allocator_options,
                             OrtAllocator** allocator) noexcept override {
    *allocator = device_allocators[memory_info->device.Id()].get();
  }

  OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) override {
    // TODO: Wrap the IDataTransfer implementation so we can copy to device using OrtApi CopyTensors.
    *data_transfer = nullptr;
    return nullptr;
  }
  */
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
