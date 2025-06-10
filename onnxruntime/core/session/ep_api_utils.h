// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
// helper to forward a call from the C API to an instance of the factory implementation.
// used by EpFactoryInternal and EpFactoryProviderBridge.
template <typename TFactory>
struct ForwardToFactory {
  static const char* ORT_API_CALL GetFactoryName(const OrtEpFactory* this_ptr) {
    return static_cast<const TFactory*>(this_ptr)->GetName();
  }

  static const char* ORT_API_CALL GetVendor(const OrtEpFactory* this_ptr) {
    return static_cast<const TFactory*>(this_ptr)->GetVendor();
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevices(OrtEpFactory* this_ptr,
                                                     const OrtHardwareDevice* const* devices,
                                                     size_t num_devices,
                                                     OrtEpDevice** ep_devices,
                                                     size_t max_ep_devices,
                                                     size_t* num_ep_devices) {
    return static_cast<TFactory*>(this_ptr)->GetSupportedDevices(devices, num_devices, ep_devices,
                                                                 max_ep_devices, num_ep_devices);
  }

  static OrtStatus* ORT_API_CALL CreateEp(OrtEpFactory* this_ptr,
                                          const OrtHardwareDevice* const* devices,
                                          const OrtKeyValuePairs* const* ep_metadata_pairs,
                                          size_t num_devices,
                                          const OrtSessionOptions* session_options,
                                          const OrtLogger* logger,
                                          OrtEp** ep) {
    return static_cast<TFactory*>(this_ptr)->CreateEp(devices, ep_metadata_pairs, num_devices,
                                                      session_options, logger, ep);
  }

  static void ORT_API_CALL ReleaseEp(OrtEpFactory* this_ptr, OrtEp* ep) {
    static_cast<TFactory*>(this_ptr)->ReleaseEp(ep);
  }
};
}  // namespace onnxruntime
