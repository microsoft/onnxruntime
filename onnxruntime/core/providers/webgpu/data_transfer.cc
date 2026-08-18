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
                                            size_t bytes) const {
  if (bytes > 0) {
    if (dst_is_gpu) {
      if (src_is_gpu) {
        // copy from GPU to GPU
        buffer_manager_.MemCpy(static_cast<WGPUBuffer>(const_cast<void*>(src_data)),
                               static_cast<WGPUBuffer>(dst_data),
                               bytes);
      } else {
        // copy from CPU to GPU
        buffer_manager_.Upload(const_cast<void*>(src_data),
                               static_cast<WGPUBuffer>(dst_data),
                               bytes);
      }
    } else {
      // copy from GPU to CPU
      buffer_manager_.Download(static_cast<WGPUBuffer>(const_cast<void*>(src_data)),
                               dst_data,
                               bytes);
    }
  }

  return Status::OK();
}

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  // WebGPU allocations always carry VendorIds::NONE. Any other vendor ID on a GPU device belongs
  // to a different provider (CUDA, DML, CANN, ...), and DataTransferManager::CopyTensor selects
  // the first registered transfer whose CanCopy accepts the pair. Without this check a session
  // registered as [WebGpuExecutionProvider, CUDAExecutionProvider] would route a CUDA copy here,
  // and CopyTensor would reinterpret the CUDA allocation as a WGPUBuffer and hand that foreign
  // handle to Dawn. The plugin data transfer already guards on vendor ID for the same reason.
  if (src_device.Type() == OrtDevice::GPU && src_device.Vendor() != OrtDevice::VendorIds::NONE) {
    return false;
  }
  if (dst_device.Type() == OrtDevice::GPU && dst_device.Vendor() != OrtDevice::VendorIds::NONE) {
    return false;
  }

  return (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::CPU) ||
         (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::GPU) ||
         (dst_device.Type() == OrtDevice::CPU && src_device.Type() == OrtDevice::GPU);
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  return impl_.CopyTensor(src.DataRaw(),
                          src.Location().device.Type() == OrtDevice::GPU,
                          dst.MutableDataRaw(),
                          dst.Location().device.Type() == OrtDevice::GPU,
                          src.SizeInBytes());
}

}  // namespace webgpu
}  // namespace onnxruntime
