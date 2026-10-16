// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/data_transfer.h"

#include <cstring>
#include <string>
#include <string_view>

#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// WebGPU requires that the copy size in CopyBufferToBuffer must be a multiple of 4 bytes.
constexpr size_t NormalizeCopySize(size_t size) {
  return (size + 3) / 4 * 4;
}

void EnforceBufferUnmapped(WebGpuContext& context, WGPUBuffer buffer) {
  if (context.ValidationMode() > ValidationMode::Basic) {
    ORT_ENFORCE(wgpuBufferGetMapState(buffer) == WGPUBufferMapState_Unmapped, "Buffer is still mapped.");
  }
}

}  // namespace

void DataTransferImpl::Upload(const void* src, WGPUBuffer dst, size_t size) const {
  // If the buffer is mapped, we can directly write to it.
  void* mapped_data = wgpuBufferGetMappedRange(dst, 0, WGPU_WHOLE_MAP_SIZE);  // ensure the buffer is mapped
  if (mapped_data) {
    memcpy(mapped_data, src, size);
    wgpuBufferUnmap(dst);
    return;
  }

  // Otherwise, we need to use a staging buffer to upload data.
  auto copy_size = NormalizeCopySize(size);

  wgpu::BufferDescriptor desc{};
  desc.size = copy_size;
  desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite;
  desc.mappedAtCreation = true;

  auto staging_buffer = context_.Device().CreateBuffer(&desc);
  mapped_data = staging_buffer.GetMappedRange();
  memcpy(mapped_data, src, size);
  // NOTE: When copy_size != size (due to 4-byte alignment requirement of CopyBufferToBuffer),
  // the trailing bytes [size, copy_size) in the staging buffer contain uninitialized data.
  // This dirty data gets copied into the destination buffer and may cause problems.
  // A possible solution is to use CopyBufferToBuffer for the aligned portion and a compute
  // shader to write the non-aligned remainder.
  staging_buffer.Unmap();

  ORT_THROW_IF_ERROR(context_.EncodeDeferredDispatches(recording_));
  auto& command_encoder = context_.GetCommandEncoder(recording_);
  context_.EndComputePass(recording_);
  command_encoder.CopyBufferToBuffer(staging_buffer, 0, dst, 0, copy_size);
  ORT_THROW_IF_ERROR(context_.Flush(buffer_manager_, recording_));
}

void DataTransferImpl::MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size) const {
  ORT_ENFORCE(src != dst, "Source and destination buffers must be different.");
  EnforceBufferUnmapped(context_, src);
  EnforceBufferUnmapped(context_, dst);

  auto copy_size = NormalizeCopySize(size);
  auto src_size = static_cast<size_t>(wgpuBufferGetSize(src));
  auto dst_size = static_cast<size_t>(wgpuBufferGetSize(dst));
  ORT_ENFORCE(copy_size <= src_size && copy_size <= dst_size,
              "Source and destination buffers must have enough space for the copy operation. src_size=",
              src_size, ", dst_size=", dst_size, ", copy_size=", copy_size, ".");

  context_.CopyBuffer(src, dst, copy_size, recording_);
}

void DataTransferImpl::Download(WGPUBuffer src, void* dst, size_t size) const {
  // Encode pending deferred dispatches before recording the readback; the flush below submits both
  // in order.
  ORT_THROW_IF_ERROR(context_.EncodeDeferredDispatches(recording_));

  EnforceBufferUnmapped(context_, src);
  auto buffer_size = (size + 15) / 16 * 16;

  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;

  auto staging_buffer = context_.Device().CreateBuffer(&desc);
  auto& command_encoder = context_.GetCommandEncoder(recording_);
  context_.EndComputePass(recording_);
  command_encoder.CopyBufferToBuffer(src, 0, staging_buffer, 0, buffer_size);
  ORT_THROW_IF_ERROR(context_.Flush(buffer_manager_, recording_));

  // TODO: revise wait in whole project

  struct MapAsyncResult {
    wgpu::MapAsyncStatus status{};
    std::string message{};
  } map_async_result;

  ORT_THROW_IF_ERROR(context_.Wait(
      staging_buffer.MapAsync(
          wgpu::MapMode::Read, 0, buffer_size, wgpu::CallbackMode::WaitAnyOnly,
          // Note: Don't throw from a Dawn callback.
          [](wgpu::MapAsyncStatus status, wgpu::StringView message,
             MapAsyncResult* result) noexcept {
            result->status = status;
            if (auto message_sv = static_cast<std::string_view>(message);
                !message_sv.empty()) {
              result->message = std::string{message_sv};
            }
          },
          &map_async_result)));

  ORT_ENFORCE(map_async_result.status == wgpu::MapAsyncStatus::Success,
              "Failed to download data from buffer. wgpu::MapAsyncStatus value: ",
              static_cast<int>(map_async_result.status), ", message: ", map_async_result.message);

  auto mapped_data = staging_buffer.GetConstMappedRange();
  memcpy(dst, mapped_data, size);
  staging_buffer.Unmap();
}

common::Status DataTransferImpl::CopyTensor(void const* src_data,
                                            bool src_is_gpu,
                                            void* dst_data,
                                            bool dst_is_gpu,
                                            size_t bytes) const {
  if (bytes > 0) {
    if (dst_is_gpu) {
      if (src_is_gpu) {
        // copy from GPU to GPU
        MemCpy(static_cast<WGPUBuffer>(const_cast<void*>(src_data)), static_cast<WGPUBuffer>(dst_data), bytes);
      } else {
        // copy from CPU to GPU
        Upload(src_data, static_cast<WGPUBuffer>(dst_data), bytes);
      }
    } else {
      // copy from GPU to CPU
      Download(static_cast<WGPUBuffer>(const_cast<void*>(src_data)), dst_data, bytes);
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

}  // namespace webgpu
}  // namespace onnxruntime
