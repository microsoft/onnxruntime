// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "gpu_data_transfer.h"
#include "migraphx_call.h"

namespace onnxruntime {
bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::GPU || src_device.MemType() == OrtDevice::MemType::HIP_PINNED || dst_device.Type() == OrtDevice::GPU || dst_device.MemType() == OrtDevice::MemType::HIP_PINNED;
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
        HIP_CALL_THROW(hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToDevice));
      }
    } else {
      // copy from other CPU memory to GPU, this is blocking
      HIP_CALL_THROW(hipMemcpy(dst_data, src_data, bytes, hipMemcpyHostToDevice));
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    // copying from GPU to CPU memory, this is blocking
    HIP_CALL_THROW(hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToHost));
  } else {
    // copying between cpu memory
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
    if (src_device.Type() == OrtDevice::CPU && src_device.MemType() == OrtDevice::MemType::HIP_PINNED) {
      // copy from pinned memory to GPU, this is non-blocking
      HIP_CALL_THROW(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyHostToDevice, static_cast<hipStream_t>(stream.GetHandle())));
    } else if (src_device.Type() == OrtDevice::GPU) {
      // copying between GPU, this is non-blocking
      HIP_CALL_THROW(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream.GetHandle())));
    } else {
      // copy from other CPU memory to GPU, this is blocking
      HIP_CALL_THROW(hipMemcpy(dst_data, src_data, bytes, hipMemcpyHostToDevice));
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
#ifndef MIGRAPHX_STREAM_SYNC
    if (dst_device.Type() == OrtDevice::CPU && dst_device.MemType() == OrtDevice::MemType::HIP_PINNED) {
      // copying from GPU to pinned memory, this is non-blocking
      HIP_CALL_THROW(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToHost, static_cast<hipStream_t>(stream.GetHandle())));
    } else {
      // copying from GPU to CPU memory, this is blocking
      HIP_CALL_THROW(hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToHost));
    }
#else
    HIP_CALL_THROW(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToHost, static_cast<hipStream_t>(stream.GetHandle())));
#endif
  } else {
    // copying between cpu memory
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}
}  // namespace onnxruntime
