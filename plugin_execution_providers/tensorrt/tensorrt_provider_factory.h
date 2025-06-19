#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#define RETURN_IF_ERROR(fn)   \
  do {                        \
    OrtStatus* status = (fn); \
    if (status != nullptr) {  \
      return status;          \
    }                         \
  } while (0)

#define RETURN_IF(cond, ort_api, msg)                    \
  do {                                                   \
    if ((cond)) {                                        \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg)); \
    }                                                    \
  } while (0)

static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr);
static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr);
static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                       const OrtHardwareDevice* const* devices,
                                                       size_t num_devices,
                                                       OrtEpDevice** ep_devices,
                                                       size_t max_ep_devices,
                                                       size_t* p_num_ep_devices);
static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                            _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                            _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                            _In_ size_t num_devices,
                                            _In_ const OrtSessionOptions* session_options,
                                            _In_ const OrtLogger* logger,
                                            _Out_ OrtEp** ep);
static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep);

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

/// <summary>
///
/// Plugin TensorRT EP factory that can create an OrtEp and return information about the supported hardware devices.
///
/// </summary>
struct TensorrtExecutionProviderFactory : OrtEpFactory, ApiPtrs {
  TensorrtExecutionProviderFactory(const char* ep_name, ApiPtrs apis) : ApiPtrs(apis), ep_name_{ep_name} {
    ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }
  const std::string ep_name_;            // EP name
  const std::string vendor_{"Nvidia"};  // EP vendor name
};