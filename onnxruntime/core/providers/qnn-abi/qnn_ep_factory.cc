// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include "qnn_ep_factory.h"

#include <cassert>

// #include "qnn_ep.h"
#include "test/autoep/library/ep_data_transfer.h"
#include "test/autoep/library/example_plugin_ep_utils.h"
#include "test/autoep/library/ep_allocator.h"
#include "core/framework/error_code_helper.h"
#include <iostream>

// Global registry to store the QNN EP factory for auto EP selection
static OrtEpFactory* g_qnn_plugin_factory = nullptr;

namespace onnxruntime {
#if !BUILD_QNN_EP_STATIC_LIB

// OrtEpApi infrastructure to be able to use the QNN EP as an OrtEpFactory for auto EP selection.
QnnEpFactory::QnnEpFactory(const char* ep_name,
            const ApiPtrs& ort_api_in,
            OrtHardwareDeviceType hw_type,
            const char* qnn_backend_type)
    : ApiPtrs(ort_api_in), ep_name_{ep_name}, ort_hw_device_type_{hw_type}, qnn_backend_type_{qnn_backend_type} {
    std::cout << "DEBUG: QnnEpFactory constructor - ep_name=" << ep_name
              << " hw_type=" << hw_type << " backend=" << qnn_backend_type << std::endl;
    ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;


    CreateDataTransfer = CreateDataTransferImpl;

}

  // Returns the name for the EP. Each unique factory configuration must have a unique name.
  // Ex: a factory that supports NPU should have a different than a factory that supports GPU.
const char* ORT_API_CALL QnnEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
    return factory->ep_name_.c_str();
}

const char* ORT_API_CALL QnnEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
    return factory->vendor_.c_str();
  }

  // Creates and returns OrtEpDevice instances for all OrtHardwareDevices that this factory supports.
  // An EP created with this factory is expected to be able to execute a model with *all* supported
  // hardware devices at once. A single instance of QNN EP is not currently setup to partition a model among
  // multiple different QNN backends at once (e.g, npu, cpu, gpu), so this factory instance is set to only
  // support one backend: npu. To support a different backend, like gpu, create a different factory instance
  // that only supports GPU.
OrtStatus* ORT_API_CALL QnnEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* p_num_ep_devices) noexcept {
    std::cout << "DEBUG: QNN GetSupportedDevicesImpl called with " << num_devices << " devices" << std::endl;
    size_t& num_ep_devices = *p_num_ep_devices;
    auto* factory = static_cast<QnnEpFactory*>(this_ptr);

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];
      auto device_type = factory->ort_api.HardwareDevice_Type(&device);
      auto vendor_id = factory->ort_api.HardwareDevice_VendorId(&device);
      std::cout << "DEBUG: Device " << i << " type=" << device_type << " vendor=" << vendor_id
                << " factory_type=" << factory->ort_hw_device_type_ << " factory_vendor=" << factory->vendor_id_ << std::endl;
      // For CPU devices, accept vendor ID 0 (generic CPU) as well as Qualcomm vendor ID
      bool vendor_match = (vendor_id == factory->vendor_id_) ||
                          (device_type == OrtHardwareDeviceType_CPU && vendor_id == 0);
      if (device_type == factory->ort_hw_device_type_ && vendor_match) {
        std::cout << "DEBUG: Device " << i << " matches! Creating EpDevice..." << std::endl;
        OrtKeyValuePairs* ep_options = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_options);
        factory->ort_api.AddKeyValuePair(ep_options, "backend_type", factory->qnn_backend_type_.c_str());
        ORT_API_RETURN_IF_ERROR(
            factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, nullptr, ep_options,
                                                        &ep_devices[num_ep_devices++]));
      }
    }

    std::cout << "DEBUG: GetSupportedDevicesImpl created " << num_ep_devices << " EpDevices" << std::endl;
    return nullptr;
  }

OrtStatus* ORT_API_CALL QnnEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                 _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                 _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                 _In_ size_t num_devices,
                                 _In_ const OrtSessionOptions* session_options,
                                 _In_ const OrtLogger* logger,
                                 _Out_ OrtEp** ep) noexcept {
    std::cout << "DEBUG: QNN CreateEpImpl called!" << std::endl;
    auto* factory = static_cast<QnnEpFactory*>(this_ptr);
    *ep = nullptr;

    if (num_devices != 1) {
        // we only registered for CPU and only expected to be selected for one CPU
        // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
        // the EP has been selected for.
        return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                            "Example EP only supports selection for one device.");
    }

    // Create the execution provider
    std::cout << "DEBUG: About to create QnnEp instance" << std::endl;
    RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                        OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                        "Creating QNN EP", ORT_FILE, __LINE__, __FUNCTION__));

    // use properties from the device and ep_metadata if needed
    // const OrtHardwareDevice* device = devices[0];
    // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

    // Create EP configuration from session options, if needed.
    // Note: should not store a direct reference to the session options object as its lifespan is not guaranteed.
    std::string ep_context_enable;
    RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(factory->ort_api, *session_options,
                                                    "ep.context_enable", "0", ep_context_enable));

    QnnEp::Config config = {};
    config.enable_ep_context = ep_context_enable == "1";

    auto dummy_ep = std::make_unique<QnnEp>(*factory, factory->ep_name_, config, logger);

    *ep = dummy_ep.release();
    return nullptr;
  }

void ORT_API_CALL QnnEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
    QnnEp* dummy_ep = static_cast<QnnEp*>(ep);
    delete dummy_ep;
  }

OrtStatus* ORT_API_CALL QnnEpFactory::CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                                 OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<QnnEpFactory*>(this_ptr);
  factory;
  std::cout << "DEBUG: QNN CreateDataTransferImpl called!" << std::endl;
  *data_transfer = nullptr; //factory.data_transfer_impl_.get();

  return nullptr;
}
};

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  std::cout << "DEBUG: QNN-ABI CreateEpFactories called!" << std::endl;
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();

  // Factory could use registration_name or define its own EP name.
  auto factory_cpu = std::make_unique<onnxruntime::QnnEpFactory>("QnnAbiTestProvider",
                                                    ApiPtrs{*ort_api, *ep_api, *model_editor_api},
                                                    OrtHardwareDeviceType_CPU,
                                                    "cpu");


  // If want to support GPU, create a new factory instance because QNN EP is not currently setup to partition a single model
  // among heterogeneous devices.
  // std::unique_ptr<OrtEpFactory> factory_gpu = std::make_unique<QnnEpFactory>(*ort_api, "QNNExecutionProvider_GPU", OrtHardwareDeviceType_GPU, "gpu");

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory_cpu.release();
  *num_factories = 1;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<onnxruntime::QnnEpFactory*>(factory);
  return nullptr;
}

}  // namespace onnxruntime

#endif  // !BUILD_QNN_EP_STATIC_LIB
