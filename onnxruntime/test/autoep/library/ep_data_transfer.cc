// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_data_transfer.h"

#include <cassert>
#include <gsl/span>

/*static*/
bool ORT_API_CALL ExampleDataTransfer::CanCopyImpl(void* this_ptr,
                                                   const OrtMemoryDevice* src_memory_device,
                                                   const OrtMemoryDevice* dst_memory_device) noexcept {
  auto& impl = *static_cast<ExampleDataTransfer*>(this_ptr);
  bool src_is_our_device = impl.ep_api.OrtMemoryDevice_AreEqual(src_memory_device, impl.device_mem_info);
  bool dst_is_our_device = impl.ep_api.OrtMemoryDevice_AreEqual(dst_memory_device, impl.device_mem_info);

  return src_is_our_device || dst_is_our_device;
}

// function to copy one or more tensors.
// implementation can optionally use async copy if a stream is available for the input.
/*static*/
OrtStatus* ORT_API_CALL ExampleDataTransfer::CopyTensorsImpl(void* this_ptr,
                                                             const OrtValue** src_tensors_ptr,
                                                             OrtValue** dst_tensors_ptr,
                                                             OrtSyncStream** streams_ptr,
                                                             size_t num_tensors) noexcept {
  auto& impl = *static_cast<ExampleDataTransfer*>(this_ptr);

  auto src_tensors = gsl::make_span<const OrtValue*>(src_tensors_ptr, num_tensors);
  auto dst_tensors = gsl::make_span<OrtValue*>(dst_tensors_ptr, num_tensors);
  auto streams = gsl::make_span<OrtSyncStream*>(streams_ptr, num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    // NOTE: Stream support will be a separate PR. ignore teh streams_ptr values for now

    const OrtMemoryDevice* src_device = nullptr;
    const OrtMemoryDevice* dst_device = nullptr;
    RETURN_IF_ERROR(impl.ep_api.OrtValue_GetMemoryDevice(src_tensors[i], &src_device));
    RETURN_IF_ERROR(impl.ep_api.OrtValue_GetMemoryDevice(dst_tensors[i], &dst_device));

    OrtMemoryInfoDeviceType src_device_type = impl.ep_api.OrtMemoryDevice_GetDeviceType(src_device);
    OrtMemoryInfoDeviceType dst_device_type = impl.ep_api.OrtMemoryDevice_GetDeviceType(dst_device);
    OrtDeviceMemoryType src_mem_type = impl.ep_api.OrtMemoryDevice_GetMemoryType(src_device);
    OrtDeviceMemoryType dst_mem_type = impl.ep_api.OrtMemoryDevice_GetMemoryType(dst_device);

    const void* src_data = nullptr;
    void* dst_data = nullptr;
    RETURN_IF_ERROR(impl.ort_api.GetTensorData(src_tensors[i], &src_data));
    RETURN_IF_ERROR(impl.ort_api.GetTensorMutableData(dst_tensors[i], &dst_data));

    // bool copy_involves_pinned_memory = src_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE ||
    //                                    dst_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE;

    if (dst_device_type == OrtMemoryInfoDeviceType_GPU) {
      if (src_device_type == OrtMemoryInfoDeviceType_GPU) {
        // GPU -> GPU
      } else {
        // CPU -> GPU
      }
    } else if (src_device_type == OrtMemoryInfoDeviceType_GPU) {
      // GPU -> CPU
    } else {
      // CPU -> CPU involves copy to/from pinned memory and a synchronize may be required first
    }
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleDataTransfer::ReleaseImpl(void* this_ptr) noexcept {
  delete static_cast<ExampleDataTransfer*>(this_ptr);
}
