// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/buffer_manager.h"

namespace onnxruntime {
namespace webgpu {

common::Status DataTransferImpl::CopyTensor(void const* src_data,
                                            bool src_is_gpu,
                                            void* dst_data,
                                            bool dst_is_gpu,
                                            size_t bytes,
                                            size_t src_offset,
                                            size_t dst_offset) const {
  if (bytes > 0) {
    if (dst_is_gpu) {
      if (src_is_gpu) {
        // copy from GPU to GPU
        buffer_manager_.MemCpy(static_cast<WGPUBuffer>(const_cast<void*>(src_data)),
                               static_cast<WGPUBuffer>(dst_data),
                               bytes, src_offset, dst_offset);
      } else {
        // copy from CPU to GPU
        buffer_manager_.Upload(const_cast<void*>(src_data),
                               static_cast<WGPUBuffer>(dst_data),
                               bytes, dst_offset);
      }
    } else {
      // copy from GPU to CPU
      buffer_manager_.Download(static_cast<WGPUBuffer>(const_cast<void*>(src_data)),
                               dst_data,
                               bytes, src_offset);
    }
  }

  return Status::OK();
}

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::CPU) ||
         (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::GPU) ||
         (dst_device.Type() == OrtDevice::CPU && src_device.Type() == OrtDevice::GPU);
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  const bool src_is_gpu = src.Location().device.Type() == OrtDevice::GPU;
  const bool dst_is_gpu = dst.Location().device.Type() == OrtDevice::GPU;
  ORT_RETURN_IF(src.ByteOffset() < 0 || dst.ByteOffset() < 0,
                "Tensor buffer offsets must not be negative.");
  return impl_.CopyTensor(
      src_is_gpu ? src.DataRawBase() : src.DataRaw(),
      src_is_gpu,
      dst_is_gpu ? dst.MutableDataRawBase() : dst.MutableDataRaw(),
      dst_is_gpu,
      src.SizeInBytes(),
      src_is_gpu ? static_cast<size_t>(src.ByteOffset()) : 0,
      dst_is_gpu ? static_cast<size_t>(dst.ByteOffset()) : 0);
}

}  // namespace webgpu
}  // namespace onnxruntime
