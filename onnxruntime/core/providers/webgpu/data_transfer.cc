// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/webgpu_context.h"

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
    void const* src_data = src.DataRaw();
    void* dst_data = dst.MutableDataRaw();

    auto& src_device = src.Location().device;
    auto& dst_device = dst.Location().device;

    if (dst_device.Type() == OrtDevice::GPU) {
      if (src_device.Type() == OrtDevice::GPU) {
        // copy from GPU to GPU
        context_.BufferManager().MemCpy(static_cast<WGPUBuffer>(const_cast<void*>(src_data)),
                                        static_cast<WGPUBuffer>(dst_data), bytes);
      } else {
        // copy from CPU to GPU
        context_.BufferManager().Upload(const_cast<void*>(src_data), static_cast<WGPUBuffer>(dst_data), bytes);
      }
    } else /* if (src_device.Type() == OrtDevice::GPU) */ {
      // copy from GPU to CPU
      context_.BufferManager().Download(static_cast<WGPUBuffer>(const_cast<void*>(src_data)), dst_data, bytes);
    }
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
