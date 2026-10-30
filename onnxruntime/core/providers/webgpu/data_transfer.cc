// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/data_transfer.h"

#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

common::Status DataTransferImpl::CopyTensor(void const* src_data,
                                            bool src_is_gpu,
                                            void* dst_data,
                                            bool dst_is_gpu,
                                            size_t bytes) const {
  std::lock_guard<std::recursive_mutex> lock{recording_.mutex};
  auto& command_state = recording_;
  if (bytes > 0) {
    if (dst_is_gpu) {
      if (src_is_gpu) {
        // copy from GPU to GPU
        buffer_manager_.MemCpy(command_state,
                               static_cast<WGPUBuffer>(const_cast<void*>(src_data)),
                               static_cast<WGPUBuffer>(dst_data),
                               bytes);
      } else {
        // copy from CPU to GPU
        buffer_manager_.Upload(command_state,
                               const_cast<void*>(src_data),
                               static_cast<WGPUBuffer>(dst_data),
                               bytes);
      }
    } else {
      // copy from GPU to CPU
      buffer_manager_.Download(command_state,
                               static_cast<WGPUBuffer>(const_cast<void*>(src_data)),
                               dst_data,
                               bytes);
    }
  }

  return Status::OK();
}

bool DataTransfer::IsSupportedDevicePair(const OrtDevice& src_device, const OrtDevice& dst_device, int context_id) {
  // WebGPU allocations carry VendorIds::NONE. A vendor-tagged GPU handle belongs to another EP, and
  // reinterpreting it as a WGPUBuffer would be unsafe. The plugin EP transfer applies the same rule.
  if (src_device.Type() == OrtDevice::GPU &&
      (src_device.Vendor() != OrtDevice::VendorIds::NONE || src_device.Id() != context_id)) {
    return false;
  }
  if (dst_device.Type() == OrtDevice::GPU &&
      (dst_device.Vendor() != OrtDevice::VendorIds::NONE || dst_device.Id() != context_id)) {
    return false;
  }

  return (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::CPU) ||
         (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::GPU) ||
         (dst_device.Type() == OrtDevice::CPU && src_device.Type() == OrtDevice::GPU);
}

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return IsSupportedDevicePair(src_device, dst_device, context_id_);
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  return impl_.CopyTensor(src.DataRaw(),
                          src.Location().device.Type() == OrtDevice::GPU,
                          dst.MutableDataRaw(),
                          dst.Location().device.Type() == OrtDevice::GPU,
                          src.SizeInBytes());
}

common::Status DataTransfer::CopyTensors(const std::vector<SrcDstPair>& src_dst_pairs) const {
  if (src_dst_pairs.size() < 2) {
    return IDataTransfer::CopyTensors(src_dst_pairs);
  }
  InlinedVector<BufferManager::UploadInfo> uploads;
  uploads.reserve(src_dst_pairs.size());
  for (const auto& pair : src_dst_pairs) {
    const auto& src = pair.src.get();
    auto& dst = pair.dst.get();
    if (pair.src_stream != nullptr || src.Location().device.Type() != OrtDevice::CPU ||
        dst.Location().device.Type() != OrtDevice::GPU) {
      return IDataTransfer::CopyTensors(src_dst_pairs);
    }
    const size_t size = src.SizeInBytes();
    auto target = static_cast<WGPUBuffer>(dst.MutableDataRaw());
    // Preserve Upload's direct write/unmap path without partially copying the list.
    if (size != 0 && wgpuBufferGetMapState(target) != WGPUBufferMapState_Unmapped) {
      return IDataTransfer::CopyTensors(src_dst_pairs);
    }
    uploads.push_back({src.DataRaw(), target, size});
  }
  buffer_manager_.UploadBatch(recording_, uploads);
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
