#include "onnxruntime_cxx_api.h"

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#define RETURN_IF_ERROR(fn)   \
  do {                        \
    OrtStatus* status = (fn); \
    if (status != nullptr) {  \
      return status;          \
    }                         \
  } while (0)

struct ApiPtrs {
  const OrtApi& ort_api;
  // const OrtEpApi& ep_api;  TODO: Add this when we flesh out the EP API.
};

struct ExampleEp : OrtEp, ApiPtrs {
  ExampleEp(ApiPtrs apis, const std::string& name, const OrtSessionOptions& session_options, const OrtLogger& logger)
      : ApiPtrs(apis), name_{name}, session_options_{session_options}, logger_{logger} {
    // Initialize the execution provider.

    auto status = ort_api.Logger_LogMessage(&logger_,
                                            OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                            ("ExampleEp has been created with name " + name_).c_str(),
                                            ORT_FILE, __LINE__, __FUNCTION__);
    // ignore status for now
    (void)status;
  }

  ~ExampleEp() {
    // Clean up the execution provider
  }

  std::string name_;
  const OrtSessionOptions& session_options_;
  const OrtLogger& logger_;
};

struct ExampleEpFactory : OrtEpFactory, ApiPtrs {
  ExampleEpFactory(const char* ep_name, ApiPtrs apis) : ApiPtrs(apis), ep_name_{ep_name} {
    ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
    return factory->ep_name_.c_str();
  }

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) {
    const auto* factory = static_cast<const ExampleEpFactory*>(this_ptr);
    return factory->vendor_.c_str();
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) {
    size_t& num_ep_devices = *p_num_ep_devices;
    auto* factory = static_cast<ExampleEpFactory*>(this_ptr);

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      // C API
      const OrtHardwareDevice& device = *devices[i];
      if (factory->ort_api.HardwareDevice_Type(&device) == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
        // these can be returned as nullptr if you have nothing to add.
        OrtKeyValuePairs* ep_metadata = nullptr;
        OrtKeyValuePairs* ep_options = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_metadata);
        factory->ort_api.CreateKeyValuePairs(&ep_options);

        // random example using made up values
        factory->ort_api.AddKeyValuePair(ep_metadata, "version", "0.1");
        factory->ort_api.AddKeyValuePair(ep_options, "run_really_fast", "true");

        // OrtEpDevice copies ep_metadata and ep_options.
        auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                                   &ep_devices[num_ep_devices++]);

        factory->ort_api.ReleaseKeyValuePairs(ep_metadata);
        factory->ort_api.ReleaseKeyValuePairs(ep_options);

        if (status != nullptr) {
          return status;
        }
      }

      // C++ API equivalent. Throws on error.
      //{
      //  Ort::ConstHardwareDevice device(devices[i]);
      //  if (device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      //    Ort::KeyValuePairs ep_metadata;
      //    Ort::KeyValuePairs ep_options;
      //    ep_metadata.Add("version", "0.1");
      //    ep_options.Add("run_really_fast", "true");
      //    Ort::EpDevice ep_device{*this_ptr, device, ep_metadata.GetConst(), ep_options.GetConst()};
      //    ep_devices[num_ep_devices++] = ep_device.release();
      //  }
      //}
    }

    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                              _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              _In_ size_t num_devices,
                                              _In_ const OrtSessionOptions* session_options,
                                              _In_ const OrtLogger* logger,
                                              _Out_ OrtEp** ep) {
    auto* factory = static_cast<ExampleEpFactory*>(this_ptr);
    *ep = nullptr;

    if (num_devices != 1) {
      // we only registered for CPU and only expected to be selected for one CPU
      // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
      // the EP has been selected for.
      return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                           "Example EP only supports selection for one device.");
    }

    // Create the execution provider
    RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                       "Creating Example EP", ORT_FILE, __LINE__, __FUNCTION__));

    // use properties from the device and ep_metadata if needed
    // const OrtHardwareDevice* device = devices[0];
    // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

    auto dummy_ep = std::make_unique<ExampleEp>(*factory, factory->ep_name_, *session_options, *logger);

    *ep = dummy_ep.release();
    return nullptr;
  }

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) {
    ExampleEp* dummy_ep = static_cast<ExampleEp*>(ep);
    delete dummy_ep;
  }

  const std::string ep_name_;            // EP name
  const std::string vendor_{"Contoso"};  // EP vendor name
};

// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define EXPORT_SYMBOL
#endif

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                           OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  // const OrtEpApi* ep_api = ort_api->GetEpApi();

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory = std::make_unique<ExampleEpFactory>(registration_name,
                                                                             ApiPtrs{*ort_api});

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete factory;
  return nullptr;
}

}  // extern "C"
