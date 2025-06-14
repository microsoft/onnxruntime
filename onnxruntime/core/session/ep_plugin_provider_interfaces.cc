// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_plugin_provider_interfaces.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/error_code_helper.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_logger.h"
#include "core/session/allocator_adapters.h"
#include "core/providers/partitioning_utils.h"

namespace onnxruntime {

//
// PluginExecutionProviderFactory
//

PluginExecutionProviderFactory::PluginExecutionProviderFactory(OrtEpFactory& ep_factory,
                                                               gsl::span<const OrtEpDevice* const> ep_devices)
    : ep_factory_{ep_factory},
      devices_{ep_devices.begin(), ep_devices.end()} {
  hardware_devices_.reserve(ep_devices.size());
  ep_metadata_.reserve(ep_devices.size());

  for (const auto* ep_device : devices_) {
    hardware_devices_.push_back(ep_device->device);
    ep_metadata_.push_back(&ep_device->ep_metadata);
  }
}

std::unique_ptr<IExecutionProvider>
PluginExecutionProviderFactory::CreateProvider(const OrtSessionOptions& session_options,
                                               const OrtLogger& session_logger) {
  OrtEp* ort_ep = nullptr;
  OrtStatus* status = ep_factory_.CreateEp(&ep_factory_, hardware_devices_.data(), ep_metadata_.data(),
                                           devices_.size(), &session_options, &session_logger, &ort_ep);
  if (status != nullptr) {
    ORT_THROW("Error creating execution provider: ", ToStatus(status).ToString());
  }

  auto ep_wrapper = std::make_unique<PluginExecutionProvider>(UniqueOrtEp(ort_ep, OrtEpDeleter(ep_factory_)),
                                                              ep_factory_,
                                                              devices_);
  ep_wrapper->SetLogger(session_logger.ToInternal());

  return ep_wrapper;
}

//
// PluginExecutionProvider
//

PluginExecutionProvider::PluginExecutionProvider(UniqueOrtEp ep, OrtEpFactory& ep_factory,
                                                 gsl::span<const OrtEpDevice* const> ep_devices)
    : IExecutionProvider(ep->GetName(ep.get()), OrtDevice()),  // TODO: What to do about OrtDevice for plugins?
      plugin_ep_(std::move(ep)),
      ep_factory_{ep_factory},
      ep_devices_{ep_devices.begin(), ep_devices.end()} {
  // generally there should only be one OrtEpDevice.
  // if there are multiple we expect them to come from the same factory.
  ORT_ENFORCE(std::all_of(ep_devices_.begin(), ep_devices_.end(),
                          [this](const OrtEpDevice* ep_device) { return ep_device->ep_factory == &ep_factory_; }));

  for (const auto* ep_device : ep_devices_) {
    if (ep_device->device_memory_info != nullptr) {
      allocator_mem_infos_.push_back(ep_device->device_memory_info);
    }

    if (ep_device->host_accessible_memory_info != nullptr) {
      allocator_mem_infos_.push_back(ep_device->host_accessible_memory_info);
    }
  }
}

std::unique_ptr<onnxruntime::IDataTransfer> PluginExecutionProvider::GetDataTransfer() const {
  OrtDataTransferImpl* data_transfer_impl = nullptr;
  OrtStatus* status = ep_factory_.CreateDataTransfer(&ep_factory_, &data_transfer_impl);
  if (status != nullptr) {
    ORT_THROW("Error creating data transfer: ", ToStatus(status).ToString());
  }

  auto dt = std::make_unique<plugin_ep::DataTransfer>(*data_transfer_impl);
  return dt;
}

std::vector<AllocatorPtr> PluginExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> allocators;
  allocators.reserve(allocator_mem_infos_.size());

  for (const auto* memory_info : allocator_mem_infos_) {
    OrtAllocator* ort_allocator_ptr = nullptr;
    OrtStatus* ort_status = ep_factory_.CreateAllocator(&ep_factory_, memory_info, nullptr, &ort_allocator_ptr);

    // throw or log? start with throw
    if (ort_status != nullptr) {
      ORT_THROW("Error creating allocator: ", ToStatus(ort_status).ToString());
    }

    auto ort_allocator = OrtAllocatorUniquePtr(
        ort_allocator_ptr,
        [this](OrtAllocator* allocator) {
          ep_factory_.ReleaseAllocator(&ep_factory_, allocator);
        });
    allocators.push_back(std::make_shared<IAllocatorImplWrappingOrtAllocator>(std::move(ort_allocator)));
  }

  return allocators;
}

PluginExecutionProvider::~PluginExecutionProvider() = default;

}  // namespace onnxruntime
