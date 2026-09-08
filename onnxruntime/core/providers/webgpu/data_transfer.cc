// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/data_transfer.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include "core/common/safeint.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

namespace {

common::Status WaitForQueue(WebGpuContext& context) {
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

}  // namespace

common::Status FlushAndWait(WebGpuContext& context, const BufferManager& buffer_manager,
                            CommandRecordingState& recording) {
  ORT_RETURN_IF_ERROR(context.Flush(buffer_manager, recording));
  return WaitForQueue(context);
}

common::Status CopyTensorWithLocalEncoder(WebGpuContext& context, const void* src_data,
                                          bool src_is_gpu, void* dst_data, bool dst_is_gpu, size_t bytes) {
  if (bytes == 0) {
    return Status::OK();
  }
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

  if (!src_is_gpu && wgpuBufferGetMapState(destination) == WGPUBufferMapState_Mapped) {
    void* mapped_data = wgpuBufferGetMappedRange(destination, 0, copy_size);
    ORT_RETURN_IF_NOT(mapped_data, "Failed to access mapped WebGPU upload buffer.");
    std::memcpy(mapped_data, src_data, bytes);
    wgpuBufferUnmap(destination);
    return Status::OK();
  }
  ORT_RETURN_IF(destination && wgpuBufferGetMapState(destination) != WGPUBufferMapState_Unmapped,
                "WebGPU copy destination must be unmapped.");

  wgpu::Buffer staging_buffer;
  if (!src_is_gpu || !dst_is_gpu) {
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = copy_size;
    descriptor.usage = src_is_gpu ? wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead
                                  : wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite;
    descriptor.mappedAtCreation = !src_is_gpu;
    staging_buffer = context.Device().CreateBuffer(&descriptor);
    ORT_RETURN_IF_NOT(staging_buffer, "Failed to create WebGPU copy staging buffer.");
    if (src_is_gpu) {
      destination = staging_buffer.Get();
    } else {
      auto* mapped_data = static_cast<uint8_t*>(staging_buffer.GetMappedRange());
      ORT_RETURN_IF_NOT(mapped_data, "Failed to map WebGPU copy staging buffer.");
      std::memcpy(mapped_data, src_data, bytes);
      std::memset(mapped_data + bytes, 0, copy_size - bytes);
      staging_buffer.Unmap();
      source = staging_buffer.Get();
    }
  }

  auto encoder = context.Device().CreateCommandEncoder();
  encoder.CopyBufferToBuffer(source, 0, destination, 0, copy_size);
  auto commands = encoder.Finish();
  context.Device().GetQueue().Submit(1, &commands);
  if (dst_is_gpu) {
    return WaitForQueue(context);
  }

  struct MapResult {
    wgpu::MapAsyncStatus status = wgpu::MapAsyncStatus::Error;
    std::string message;
  } map_result;
  ORT_RETURN_IF_ERROR(context.Wait(staging_buffer.MapAsync(
      wgpu::MapMode::Read, 0, copy_size, wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::MapAsyncStatus status, wgpu::StringView message, MapResult* result) noexcept {
        result->status = status;
        if (auto text = static_cast<std::string_view>(message); !text.empty()) {
          result->message = text;
        }
      },
      &map_result)));
  ORT_RETURN_IF_NOT(map_result.status == wgpu::MapAsyncStatus::Success,
                    "WebGPU copy readback failed: ", map_result.message);
  std::memcpy(dst_data, staging_buffer.GetConstMappedRange(), bytes);
  staging_buffer.Unmap();
  return Status::OK();
}

common::Status DataTransferImpl::CopyTensor(void const* src_data,
                                            bool src_is_gpu,
                                            void* dst_data,
                                            bool dst_is_gpu,
                                            size_t bytes) const {
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
