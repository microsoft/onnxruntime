// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn-abi/qnn_provider_factory_simulation.h"

#include <cassert>
#include <iostream>

#include "core/framework/error_code_helper.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {

// OrtEpApi infrastructure to be able to use the QNN EP as an OrtEpFactory for auto EP selection.
QnnEpFactorySimulation::QnnEpFactorySimulation(const char* ep_name, ApiPtrs ort_api_in)
    : OrtEpFactory{}, ApiPtrs(ort_api_in), ep_name_{ep_name} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
}

// Returns the name for the EP. Each unique factory configuration must have a unique name.
// Ex: a factory that supports NPU should have a different than a factory that supports GPU.
const char* ORT_API_CALL QnnEpFactorySimulation::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactorySimulation*>(this_ptr);
  return factory->ep_name_.c_str();
}

const char* ORT_API_CALL QnnEpFactorySimulation::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactorySimulation*>(this_ptr);
  return factory->vendor_.c_str();
}

uint32_t ORT_API_CALL QnnEpFactorySimulation::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactorySimulation*>(this_ptr);
  return factory->vendor_id_;
}

const char* ORT_API_CALL QnnEpFactorySimulation::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactorySimulation*>(this_ptr);
  return factory->ep_version_.c_str();
}

// Creates and returns OrtEpDevice instances for all OrtHardwareDevices that this factory supports.
// An EP created with this factory is expected to be able to execute a model with *all* supported
// hardware devices at once. A single instance of QNN EP is not currently setup to partition a model among
// multiple different QNN backends at once (e.g, npu, cpu, gpu), so this factory instance is set to only
// support one backend: npu. To support a different backend, like gpu, create a different factory instance
// that only supports GPU.
OrtStatus* ORT_API_CALL QnnEpFactorySimulation::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                        const OrtHardwareDevice* const* devices,
                                                                        size_t num_devices,
                                                                        OrtEpDevice** ep_devices,
                                                                        size_t max_ep_devices,
                                                                        size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<QnnEpFactorySimulation*>(this_ptr);

  for (size_t idx = 0; idx < num_devices && num_ep_devices < max_ep_devices; ++idx) {
    const OrtHardwareDevice& device = *devices[idx];
    auto device_type = factory->ort_api.HardwareDevice_Type(&device);
    auto vendor_id = factory->ort_api.HardwareDevice_VendorId(&device);

    if (vendor_id == factory->vendor_id_ || device_type == OrtHardwareDeviceType_CPU) {
      RETURN_IF_NOT_NULL(factory->ep_api.CreateEpDevice(factory,
                                                        &device,
                                                        nullptr,
                                                        nullptr,
                                                        &ep_devices[num_ep_devices++]));
    }
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEpFactorySimulation::CreateEpImpl(OrtEpFactory* /*this_ptr*/,
                                                             _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                                             _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                             _In_ size_t /*num_devices*/,
                                                             _In_ const OrtSessionOptions* /*session_options*/,
                                                             _In_ const OrtLogger* /*logger*/,
                                                             _Out_ OrtEp** /*ep*/) noexcept {
  return nullptr;
}

void ORT_API_CALL QnnEpFactorySimulation::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  if (ep == nullptr) {
    return;
  }

  QnnEp* dummy_ep = static_cast<QnnEp*>(ep);
  delete dummy_ep;
}

OrtStatus* ORT_API_CALL QnnEpFactorySimulation::CreateDataTransferImpl(OrtEpFactory* /* this_ptr */,
                                                                       OrtDataTransferImpl** data_transfer) noexcept {
  *data_transfer = nullptr;

  return nullptr;
}

bool ORT_API_CALL QnnEpFactorySimulation::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return false;
}

}  // namespace onnxruntime

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* registration_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* /*default_logger*/,
                             OrtEpFactory** factories,
                             size_t max_factories,
                             size_t* num_factories) {
  if (ort_api_base == nullptr) {
    return nullptr;  // Cannot create status without API base
  }

  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  if (ort_api == nullptr) {
    return nullptr;  // Cannot create status without ORT API
  }

  // Manual init for the C++ API
  Ort::InitApi(ort_api);

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  if (factories == nullptr || num_factories == nullptr) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Invalid arguments: factories and num_factories cannot be null.");
  }

  const OrtEpApi* ep_api = ort_api->GetEpApi();
  if (ep_api == nullptr) {
    return ort_api->CreateStatus(ORT_FAIL, "Failed to get EP API.");
  }

  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();
  if (model_editor_api == nullptr) {
    return ort_api->CreateStatus(ORT_FAIL, "Failed to get Model Editor API.");
  }

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<onnxruntime::QnnEpFactorySimulation> factory;
  if (registration_name == nullptr) {
    registration_name = "QnnAbiTestProvider";
  }
  try {
    factory = std::make_unique<onnxruntime::QnnEpFactorySimulation>(registration_name,
                                                                    onnxruntime::ApiPtrs{*ort_api,
                                                                                         *ep_api,
                                                                                         *model_editor_api});
  } catch (const std::exception& e) {
    return ort_api->CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return ort_api->CreateStatus(ORT_FAIL, "Unknown exception occurred while creating QNN EP factory.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  if (factory == nullptr) {
    return nullptr;
  }

  delete static_cast<onnxruntime::QnnEpFactorySimulation*>(factory);
  return nullptr;
}
}
