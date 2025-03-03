// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/migraphx/gpu_data_transfer.h"
#include "core/providers/migraphx/migraphx_call.h"

// If you make change below, please also update onnxruntime/core/providers/rocm/gpu_data_transfer.cc

namespace onnxruntime {

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::GPU || src_device.MemType() == OrtDevice::MemType::HIP_PINNED ||
         dst_device.Type() == OrtDevice::GPU || dst_device.MemType() == OrtDevice::MemType::HIP_PINNED;
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  // for the sync version of memcpy, launch to hip default stream
  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::GPU) {
      // Copy only if the two addresses are different.
      if (dst_data != src_data) {
        HIP_RETURN_IF_ERROR(hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToDevice));
        // Follow core/providers/cuda/gpu_data_transfer.cc to synchronize the default stream here.
        HIP_RETURN_IF_ERROR(hipStreamSynchronize(nullptr));
      }
    } else {
      // copy from other CPU memory to GPU, this is blocking
      HIP_RETURN_IF_ERROR(hipMemcpy(dst_data, src_data, bytes, hipMemcpyHostToDevice));
      if (src_device.MemType() != OrtDevice::MemType::HIP_PINNED) {
        // Follow core/providers/cuda/gpu_data_transfer.cc to synchronize the default stream here.
        HIP_RETURN_IF_ERROR(hipStreamSynchronize(nullptr));
      }
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    // copying from GPU to CPU memory, this is blocking
    HIP_RETURN_IF_ERROR(hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToHost));
  } else {
    // copying between cpu memory
    ORT_ENFORCE(dst_data != src_data);
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

common::Status GPUDataTransfer::CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::CPU) {
      // If source are not pinned, the memory copy will be performed synchronously.
      // For best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyHostToDevice, static_cast<hipStream_t>(stream.GetHandle())));
    } else if (src_device.Type() == OrtDevice::GPU) {
      // copying between GPU, this is non-blocking
      HIP_CALL_THROW(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream.GetHandle())));
    } else {
      // copy from other CPU memory to GPU, this is blocking
      HIP_CALL_THROW(hipMemcpyWithStream(dst_data, src_data, bytes, hipMemcpyHostToDevice, static_cast<hipStream_t>(stream.GetHandle())));
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    // If dest are not pinned, the memory copy will be performed synchronously.
    // For best performance, use hipHostMalloc to allocate host memory that is transferred asynchronously.
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToHost, static_cast<hipStream_t>(stream.GetHandle())));
  } else {
    if (src_device.MemType() == OrtDevice::MemType::CUDA_PINNED) {
      // sync the stream first to make sure the data arrived
      HIP_RETURN_IF_ERROR(hipStreamSynchronize(static_cast<hipStream_t>(stream.GetHandle())));
    }
    ORT_ENFORCE(dst_data != src_data);
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

}  // namespace onnxruntime
