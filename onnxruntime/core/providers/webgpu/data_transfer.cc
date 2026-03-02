// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/buffer_manager.h"

namespace onnxruntime {
namespace webgpu {

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::CPU) ||
         (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::GPU) ||
         (dst_device.Type() == OrtDevice::CPU && src_device.Type() == OrtDevice::GPU);
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  size_t bytes = src.SizeInBytes();
  if (bytes > 0) {
    auto& src_device = src.Location().device;
    auto& dst_device = dst.Location().device;

    // For WebGPU tensors, p_data_ is an opaque WGPUBuffer handle, not an addressable pointer.
    // Tensor::DataRaw() computes (char*)p_data_ + byte_offset_, which produces a pointer-sized
    // value that encodes the buffer handle *plus* the byte offset -- not a valid WGPUBuffer.
    // We must recover the base handle by subtracting ByteOffset() back out, then pass it
    // along with the byte offset to the buffer manager so the correct region is copied.
    if (dst_device.Type() == OrtDevice::GPU) {
      const ptrdiff_t dst_byte_offset = dst.ByteOffset();
      void* dst_raw = dst.MutableDataRaw();  // = (char*)dst_wgpu_handle + dst_byte_offset
      WGPUBuffer dst_buf = reinterpret_cast<WGPUBuffer>(static_cast<char*>(dst_raw) - dst_byte_offset);

      if (src_device.Type() == OrtDevice::GPU) {
        // copy from GPU to GPU
        const ptrdiff_t src_byte_offset = src.ByteOffset();
        void const* src_raw = src.DataRaw();  // = (char*)src_wgpu_handle + src_byte_offset
        WGPUBuffer src_buf = reinterpret_cast<WGPUBuffer>(
            static_cast<char*>(const_cast<void*>(src_raw)) - src_byte_offset);
        buffer_manager_.MemCpy(src_buf, dst_buf, bytes,
                               static_cast<size_t>(src_byte_offset),
                               static_cast<size_t>(dst_byte_offset));
      } else {
        // copy from CPU to GPU
        // src.DataRaw() for a CPU tensor is a real addressable pointer (byte_offset_ already applied).
        void const* src_data = src.DataRaw();
        buffer_manager_.Upload(const_cast<void*>(src_data), dst_buf, bytes,
                               0, static_cast<size_t>(dst_byte_offset));
      }
    } else /* if (src_device.Type() == OrtDevice::GPU) */ {
      // copy from GPU to CPU
      const ptrdiff_t src_byte_offset = src.ByteOffset();
      void const* src_raw = src.DataRaw();  // = (char*)src_wgpu_handle + src_byte_offset
      WGPUBuffer src_buf = reinterpret_cast<WGPUBuffer>(
          static_cast<char*>(const_cast<void*>(src_raw)) - src_byte_offset);
      // dst.MutableDataRaw() for a CPU tensor is a real addressable pointer.
      void* dst_data = dst.MutableDataRaw();
      buffer_manager_.Download(src_buf, dst_data, bytes,
                               static_cast<size_t>(src_byte_offset), 0);
    }  // else: GPU-to-CPU
  }  // if (bytes > 0)

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
