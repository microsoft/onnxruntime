// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_ep_data_transfer.h"

#include <cassert>
#include <gsl/span>

namespace onnxruntime {

/*static*/
bool ORT_API_CALL QnnDataTransfer::CanCopyImpl(void* this_ptr,
                                               const OrtMemoryDevice* src_memory_device,
                                               const OrtMemoryDevice* dst_memory_device) noexcept {
  static constexpr uint32_t VendorId = 0x5143;

  auto& impl = *static_cast<QnnDataTransfer*>(this_ptr);
  bool src_is_our_device = impl.ep_api.MemoryDevice_AreEqual(src_memory_device, impl.device_mem_info);
  bool dst_is_our_device = impl.ep_api.MemoryDevice_AreEqual(dst_memory_device, impl.device_mem_info);

  if (src_is_our_device && dst_is_our_device) {
    return true;
  }

  // implementation should check if the copy is possible, which may require checking the device type as needed.
  OrtMemoryInfoDeviceType src_device_type = impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
  OrtMemoryInfoDeviceType dst_device_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);

  if (src_is_our_device) {
    // check device type and vendor to see if compatible
    return (dst_device_type == OrtMemoryInfoDeviceType_NPU);
  }

  if (dst_is_our_device) {
    // check device type and vendor to see if compatible
    return (src_device_type == OrtMemoryInfoDeviceType_NPU);
  }

  return false;
}

// function to copy one or more tensors.
// implementation can optionally use async copy if a stream is available for the input.
/*static*/
OrtStatus* ORT_API_CALL QnnDataTransfer::CopyTensorsImpl(void* this_ptr,
                                                         const OrtValue** src_tensors_ptr,
                                                         OrtValue** dst_tensors_ptr,
                                                         OrtSyncStream** streams_ptr,
                                                         size_t num_tensors) noexcept {
  auto& impl = *static_cast<QnnDataTransfer*>(this_ptr);

  auto src_tensors = gsl::make_span<const OrtValue*>(src_tensors_ptr, num_tensors);
  auto dst_tensors = gsl::make_span<OrtValue*>(dst_tensors_ptr, num_tensors);
  auto streams = gsl::make_span<OrtSyncStream*>(streams_ptr, num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    // NOTE: Stream support will be a separate PR. ignore teh streams_ptr values for now

    const OrtMemoryDevice* src_device = nullptr;
    const OrtMemoryDevice* dst_device = nullptr;
    RETURN_IF_ERROR(impl.ep_api.Value_GetMemoryDevice(src_tensors[i], &src_device));
    RETURN_IF_ERROR(impl.ep_api.Value_GetMemoryDevice(dst_tensors[i], &dst_device));

    OrtMemoryInfoDeviceType src_device_type = impl.ep_api.MemoryDevice_GetDeviceType(src_device);
    OrtMemoryInfoDeviceType dst_device_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_device);

    const void* src_data = nullptr;
    void* dst_data = nullptr;
    RETURN_IF_ERROR(impl.ort_api.GetTensorData(src_tensors[i], &src_data));
    RETURN_IF_ERROR(impl.ort_api.GetTensorMutableData(dst_tensors[i], &dst_data));
    size_t src_bytes;
    RETURN_IF_ERROR(impl.ort_api.GetTensorSizeInBytes(src_tensors[i], &src_bytes));

    if (dst_device_type == OrtMemoryInfoDeviceType_NPU) {
      if (src_device_type == OrtMemoryInfoDeviceType_NPU) {
        // NPU -> NPU
        // Bypass it
      } else {
        // CPU -> NPU
        // Bypass it
      }
    } else if (src_device_type == OrtMemoryInfoDeviceType_NPU) {
      // NPU -> CPU
      // Bypass it
    } else {
      // CPU -> CPU
      // Bypass it
    }
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL QnnDataTransfer::ReleaseImpl(void* /*this_ptr*/) noexcept {
  // In our setup the factory owns a shared QnnDataTransfer instance so it will do the cleanup, and we ignore
  // the call to Release from the plugin_ep::DataTransfer dtor (see /onnxruntime/core/framework/plugin_data_transfer.h)
  //
  // If you create a new instance on each call to OrtEpFactory::CreateDataTransfer you call `delete` here
  // delete static_cast<QnnDataTransfer*>(this_ptr);
}

}  // namespace onnxruntime
