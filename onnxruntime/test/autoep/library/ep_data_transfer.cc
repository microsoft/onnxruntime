// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_data_transfer.h"

#include <cassert>
#include <gsl/span>

/*static*/
bool ORT_API_CALL ExampleDataTransfer::CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                                   const OrtMemoryDevice* src_memory_device,
                                                   const OrtMemoryDevice* dst_memory_device) noexcept {
  const auto& impl = *static_cast<const ExampleDataTransfer*>(this_ptr);
  bool src_is_our_device = impl.ep_api.MemoryDevice_AreEqual(src_memory_device, impl.device_mem_info);
  bool dst_is_our_device = impl.ep_api.MemoryDevice_AreEqual(dst_memory_device, impl.device_mem_info);

  if (src_is_our_device && dst_is_our_device) {
    return true;
  }

  // implementation should check if the copy is possible, which may require checking the device type, the memory type
  // and the vendor and device IDs as needed.
  OrtMemoryInfoDeviceType src_device_type = impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
  OrtMemoryInfoDeviceType dst_device_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);
  OrtDeviceMemoryType src_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(src_memory_device);
  OrtDeviceMemoryType dst_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(dst_memory_device);

  // we can copy to/from CPU or CPU accessible memory
  if (src_is_our_device) {
    return (dst_device_type == OrtMemoryInfoDeviceType_CPU || dst_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE);
  }

  if (dst_is_our_device) {
    return (src_device_type == OrtMemoryInfoDeviceType_CPU || src_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE);
  }

  return false;
}

namespace {
void CopyImpl(const void* src_data, void* dst_data, size_t bytes, OrtSyncStream* stream) {
  // in our example setup this is really CPU to CPU

  if (stream) {
    // EP can do an async copy using the stream. e.g. an NVIDIA EP would provide the stream to cudaMemcpyAsync
  }

  memcpy(dst_data, src_data, bytes);
}
}  // namespace

// function to copy one or more tensors.
// implementation can optionally use async copy if a stream is available for the input.
/*static*/
OrtStatus* ORT_API_CALL ExampleDataTransfer::CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                             const OrtValue** src_tensors_ptr,
                                                             OrtValue** dst_tensors_ptr,
                                                             OrtSyncStream** streams_ptr,
                                                             size_t num_tensors) noexcept {
  auto& impl = *static_cast<ExampleDataTransfer*>(this_ptr);

  auto src_tensors = gsl::make_span<const OrtValue*>(src_tensors_ptr, num_tensors);
  auto dst_tensors = gsl::make_span<OrtValue*>(dst_tensors_ptr, num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    // the implementation for a 'real' EP would be something along these lines.
    // See CudaDataTransferImpl in onnxruntime\core\providers\cuda\cuda_provider_factory.cc
    const OrtMemoryDevice* src_device = nullptr;
    const OrtMemoryDevice* dst_device = nullptr;
    RETURN_IF_ERROR(impl.ep_api.Value_GetMemoryDevice(src_tensors[i], &src_device));
    RETURN_IF_ERROR(impl.ep_api.Value_GetMemoryDevice(dst_tensors[i], &dst_device));

    OrtMemoryInfoDeviceType src_device_type = impl.ep_api.MemoryDevice_GetDeviceType(src_device);
    OrtMemoryInfoDeviceType dst_device_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_device);

    //  OrtDeviceMemoryType src_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(src_device);
    //  OrtDeviceMemoryType dst_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(dst_device);
    //  bool copy_involves_host_accessible_memory = src_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE ||
    //                                              dst_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE;

    const void* src_data = nullptr;
    void* dst_data = nullptr;
    size_t bytes;

    RETURN_IF_ERROR(impl.ort_api.GetTensorData(src_tensors[i], &src_data));
    RETURN_IF_ERROR(impl.ort_api.GetTensorMutableData(dst_tensors[i], &dst_data));
    RETURN_IF_ERROR(impl.ort_api.GetTensorSizeInBytes(src_tensors[i], &bytes));

    if (dst_device_type == OrtMemoryInfoDeviceType_GPU) {
      if (src_device_type == OrtMemoryInfoDeviceType_GPU) {
        // GPU -> GPU
      } else {
        // CPU -> GPU
      }
    } else if (src_device_type == OrtMemoryInfoDeviceType_GPU) {
      // GPU -> CPU
    } else {
      // CPU -> CPU. may involve copy a to/from host accessible memory and a synchronize may be required first
    }

    // but in our example EP it's simpler as it's really a (fake) CPU to CPU copy
    CopyImpl(src_data, dst_data, bytes, streams_ptr ? streams_ptr[i] : nullptr);
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleDataTransfer::ReleaseImpl(OrtDataTransferImpl* /*this_ptr*/) noexcept {
  // In our setup the factory owns a shared ExampleDataTransfer instance so it will do the cleanup, and we ignore
  // the call to Release from the plugin_ep::DataTransfer dtor (see /onnxruntime/core/framework/plugin_data_transfer.h)
  //
  // If you create a new instance on each call to OrtEpFactory::CreateDataTransfer you call `delete` here
  // delete static_cast<ExampleDataTransfer*>(this_ptr);
}
