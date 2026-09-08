// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/data_transfer.h"

#include "core/common/safeint.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

common::Status FlushAndWait(WebGpuContext& context, const BufferManager& buffer_manager,
                            CommandRecordingState& recording) {
  ORT_RETURN_IF_ERROR(context.Flush(buffer_manager, recording));
  wgpu::QueueWorkDoneStatus completion = wgpu::QueueWorkDoneStatus::Error;
  auto future = context.Device().GetQueue().OnSubmittedWorkDone(
      wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::QueueWorkDoneStatus status, wgpu::StringView, wgpu::QueueWorkDoneStatus* result) noexcept {
        *result = status;
      },
      &completion);
  ORT_RETURN_IF_ERROR(context.Wait(future));
  ORT_RETURN_IF_NOT(completion == wgpu::QueueWorkDoneStatus::Success, "WebGPU queue completion failed.");
  return Status::OK();
}

common::Status DataTransferImpl::CopyTensor(void const* src_data,
                                            bool src_is_gpu,
                                            void* dst_data,
                                            bool dst_is_gpu,
                                            size_t bytes) const {
  auto& command_state = recording_;
  if (bytes > 0) {
    ORT_RETURN_IF_NOT(src_is_gpu || dst_is_gpu, "Expected a WebGPU copy endpoint.");
    ORT_RETURN_IF_NOT(src_data && dst_data, "WebGPU copy buffers must not be null.");
    const size_t copy_size = (SafeInt<size_t>(bytes) + 3) / 4 * 4;
    WGPUBuffer source = src_is_gpu ? static_cast<WGPUBuffer>(const_cast<void*>(src_data)) : nullptr;
    WGPUBuffer destination = dst_is_gpu ? static_cast<WGPUBuffer>(dst_data) : nullptr;
    ORT_RETURN_IF(source && copy_size > wgpuBufferGetSize(source), "WebGPU copy exceeds source buffer size.");
    ORT_RETURN_IF(destination && copy_size > wgpuBufferGetSize(destination), "WebGPU copy exceeds destination buffer size.");
    ORT_RETURN_IF(source && source == destination, "Source and destination buffers must be different.");
    ORT_RETURN_IF(source && wgpuBufferGetMapState(source) != WGPUBufferMapState_Unmapped,
                  "WebGPU copy source must be unmapped.");
    if (destination) {
      const auto map_state = wgpuBufferGetMapState(destination);
      ORT_RETURN_IF_NOT(map_state == WGPUBufferMapState_Unmapped ||
                            (!src_is_gpu && map_state == WGPUBufferMapState_Mapped),
                        "WebGPU copy destination must be unmapped.");
    }
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

bool DataTransfer::IsSupportedDevicePair(const OrtDevice& src_device, const OrtDevice& dst_device) {
  // WebGPU allocations carry VendorIds::NONE. A vendor-tagged GPU handle belongs to another EP, and
  // reinterpreting it as a WGPUBuffer would be unsafe. The plugin EP transfer applies the same rule.
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

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return IsSupportedDevicePair(src_device, dst_device);
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
