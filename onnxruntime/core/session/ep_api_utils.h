// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
// helper to forward a call from the C API to an instance of the factory implementation.
// used by EpFactoryInternal and EpFactoryProviderBridge.
template <typename TFactory>
struct ForwardToFactory {
  static const char* ORT_API_CALL GetFactoryName(const OrtEpFactory* this_ptr) noexcept {
    return static_cast<const TFactory*>(this_ptr)->GetName();
  }

  static const char* ORT_API_CALL GetVendor(const OrtEpFactory* this_ptr) noexcept {
    return static_cast<const TFactory*>(this_ptr)->GetVendor();
  }

  static const char* ORT_API_CALL GetVersion(const OrtEpFactory* this_ptr) noexcept {
    return static_cast<const TFactory*>(this_ptr)->GetVersion();
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevices(OrtEpFactory* this_ptr,
                                                     const OrtHardwareDevice* const* devices,
                                                     size_t num_devices,
                                                     OrtEpDevice** ep_devices,
                                                     size_t max_ep_devices,
                                                     size_t* num_ep_devices) noexcept {
    return static_cast<TFactory*>(this_ptr)->GetSupportedDevices(devices, num_devices, ep_devices,
                                                                 max_ep_devices, num_ep_devices);
  }

  static OrtStatus* ORT_API_CALL CreateEp(OrtEpFactory* this_ptr,
                                          const OrtHardwareDevice* const* devices,
                                          const OrtKeyValuePairs* const* ep_metadata_pairs,
                                          size_t num_devices,
                                          const OrtSessionOptions* session_options,
                                          const OrtLogger* logger,
                                          OrtEp** ep) noexcept {
    return static_cast<TFactory*>(this_ptr)->CreateEp(devices, ep_metadata_pairs, num_devices,
                                                      session_options, logger, ep);
  }

  static OrtStatus* ORT_API_CALL CreateAllocator(_In_ OrtEpFactory* this_ptr,
                                                 _In_ const OrtMemoryInfo* memory_info,
                                                 _In_opt_ const OrtEp* ep,
                                                 _In_opt_ const OrtKeyValuePairs* allocator_options,
                                                 _Outptr_ OrtAllocator** allocator) noexcept {
    return static_cast<TFactory*>(this_ptr)->CreateAllocator(memory_info, ep, allocator_options, allocator);
  }

  static void ORT_API_CALL ReleaseAllocator(_In_ OrtEpFactory* this_ptr, _In_ OrtAllocator* allocator) noexcept {
    return static_cast<TFactory*>(this_ptr)->ReleaseAllocator(allocator);
  }

  static OrtStatus* ORT_API_CALL CreateDataTransfer(_In_ OrtEpFactory* this_ptr,
                                                    _Outptr_opt_ OrtDataTransferImpl** data_transfer) noexcept {
    return static_cast<TFactory*>(this_ptr)->CreateDataTransfer(data_transfer);
  }

  static bool ORT_API_CALL IsStreamAware(_In_ const OrtEpFactory* this_ptr) noexcept {
    return static_cast<const TFactory*>(this_ptr)->IsStreamAware();
  }

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDevice(_In_ OrtEpFactory* this_ptr,
                                                           _In_ const OrtMemoryDevice* memory_device,
                                                           _In_opt_ const OrtEp* /*ep*/,
                                                           _In_opt_ const OrtKeyValuePairs* stream_options,
                                                           _Outptr_opt_ OrtSyncStreamImpl** stream) noexcept {
    // ignore the OrtEp input as we won't ever have one for internal EPs
    return static_cast<TFactory*>(this_ptr)->CreateSyncStreamForDevice(memory_device, stream_options, stream);
  }

  static void ORT_API_CALL ReleaseEp(OrtEpFactory* this_ptr, OrtEp* ep) noexcept {
    static_cast<TFactory*>(this_ptr)->ReleaseEp(ep);
  }
};
}  // namespace onnxruntime
