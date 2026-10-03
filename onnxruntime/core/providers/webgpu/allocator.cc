// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <utility>

#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"

#if defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/ep/sync_stream.h"
#endif

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

// Streamless allocation, e.g., application CreateTensor/Alloc APIs using a Session allocator,
// or framework allocations without a stream, including during Run. The plugin's writable device
// allocator submits cached clears before returning: the consumer may use a different recording.
// Built-in callers can supply a different submission policy.
void* GpuBufferAllocator::Alloc(size_t size) {
#if defined(ORT_USE_EP_API_ADAPTERS)
  if (ep::UseLegacyRecording() && recording_getter_) {
    // Legacy uses the shared recording. The Session policy defers during Run; Env always submits.
    // An omitted policy does not request submission, matching the built-in path.
    return Allocate(size, recording_getter_(),
                    should_submit_zero_initialize_ && should_submit_zero_initialize_());
  }
  // Streamless clears are independent of Run and must not read its capture state.
  CommandRecordingState recording;
  return Allocate(size, recording, true);
#else
  return Allocate(size, recording_getter_(), should_submit_zero_initialize_ && should_submit_zero_initialize_());
#endif
}

void* GpuBufferAllocator::Allocate(size_t size, CommandRecordingState& recording, bool submit_zero_initialize) {
  if (size == 0) {
    return nullptr;
  }

  wgpu::BufferUsage usage = mapped_at_creation_ ? wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapWrite
                                                : wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;

  auto buffer = buffer_manager_getter_().Create(recording, size, usage, initialize_to_zero_,
                                                submit_zero_initialize);
  num_allocs_.fetch_add(1, std::memory_order_relaxed);
  return buffer;
}

void GpuBufferAllocator::Free(void* p) {
  if (p != nullptr) {
    buffer_manager_getter_().Release(static_cast<WGPUBuffer>(p),
                                     recording_getter_ ? &recording_getter_() : nullptr);
    num_allocs_.fetch_sub(1, std::memory_order_relaxed);
  }
}

void GpuBufferAllocator::GetStats(AllocatorStats* stats) {
  *stats = AllocatorStats{};
  stats->num_allocs = num_allocs_.load(std::memory_order_relaxed);
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
