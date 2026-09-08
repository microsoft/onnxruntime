// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <utility>

#include "core/common/safeint.h"
#include "core/framework/session_state.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

GpuBufferAllocator::GpuBufferAllocator(
    std::function<const BufferManager&()> buffer_manager_getter,
    std::function<CommandRecordingState&()> recording_getter,
    bool is_read_only_allocator,
    std::function<bool()> should_submit_zero_initialize)
    : IAllocator(
          OrtMemoryInfo(WEBGPU_BUFFER,
                        is_read_only_allocator ? OrtAllocatorType::OrtReadOnlyAllocator
                                               : OrtAllocatorType::OrtDeviceAllocator,
                        WebGpuDevice,
                        OrtMemTypeDefault)),
      buffer_manager_getter_{std::move(buffer_manager_getter)},
      recording_getter_{std::move(recording_getter)},
      should_submit_zero_initialize_{std::move(should_submit_zero_initialize)},
      mapped_at_creation_{is_read_only_allocator && buffer_manager_getter_().SupportsUMA()},
      initialize_to_zero_{!is_read_only_allocator} {
}

void* GpuBufferAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  auto& recording = recording_getter_();
  stats_.num_allocs++;

  wgpu::BufferUsage usage = mapped_at_creation_ ? wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapWrite
                                                : wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;

  const bool submit_zero_initialize = should_submit_zero_initialize_ && should_submit_zero_initialize_();
  return buffer_manager_getter_().Create(recording, size, usage, initialize_to_zero_,
                                         submit_zero_initialize);
}

void GpuBufferAllocator::Free(void* p) {
  if (p != nullptr) {
    auto& recording = recording_getter_();
    buffer_manager_getter_().Release(recording, static_cast<WGPUBuffer>(p));
    stats_.num_allocs--;
  }
}

void GpuBufferAllocator::GetStats(AllocatorStats* stats) {
  *stats = stats_;
}

ExternalGpuBufferAllocator::ExternalGpuBufferAllocator(std::shared_ptr<WebGpuContext> context)
    : IAllocator(OrtMemoryInfo(WEBGPU_BUFFER,
                               OrtAllocatorType::OrtDeviceAllocator,
                               WebGpuDevice,
                               OrtMemTypeDefault)),
      context_{std::move(context)} {
}

ExternalGpuBufferAllocator::~ExternalGpuBufferAllocator() = default;

void* ExternalGpuBufferAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock{mutex_};
  wgpu::BufferDescriptor descriptor{};
  descriptor.size = (SafeInt<size_t>(size) + 15) / 16 * 16;
  descriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                     wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;
  auto buffer = context_->Device().CreateBuffer(&descriptor);
  ORT_ENFORCE(buffer, "Failed to create external WebGPU buffer: size=", size, ".");
  ++stats_.num_allocs;
  return buffer.MoveToCHandle();
}

void ExternalGpuBufferAllocator::Free(void* p) {
  if (p == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock{mutex_};
  wgpuBufferRelease(static_cast<WGPUBuffer>(p));
  --stats_.num_allocs;
}

void ExternalGpuBufferAllocator::GetStats(AllocatorStats* stats) {
  std::lock_guard<std::mutex> lock{mutex_};
  *stats = stats_;
}

WebGpuNoOpAllocator::WebGpuNoOpAllocator(bool is_read_only_allocator)
    : IAllocator(
          OrtMemoryInfo(WEBGPU_BUFFER,
                        is_read_only_allocator ? OrtAllocatorType::OrtReadOnlyAllocator
                                               : OrtAllocatorType::OrtDeviceAllocator,
                        WebGpuDevice,
                        OrtMemTypeDefault)) {
}

void* WebGpuNoOpAllocator::Alloc(size_t /*size*/) {
  ORT_THROW("WebGPU EP device-free context must not allocate device memory.");
}

void WebGpuNoOpAllocator::Free(void* /*p*/) {
}

AllocatorPtr CreateWebGpuAllocator(bool device_free,
                                   std::function<const BufferManager&()> buffer_manager_getter,
                                   std::function<CommandRecordingState&()> recording_getter,
                                   bool is_read_only_allocator,
                                   std::function<bool()> should_submit_zero_initialize) {
  if (device_free) {
    return std::make_shared<WebGpuNoOpAllocator>(is_read_only_allocator);
  }
  return std::make_shared<GpuBufferAllocator>(std::move(buffer_manager_getter), std::move(recording_getter),
                                              is_read_only_allocator,
                                              std::move(should_submit_zero_initialize));
}

}  // namespace webgpu
}  // namespace onnxruntime
