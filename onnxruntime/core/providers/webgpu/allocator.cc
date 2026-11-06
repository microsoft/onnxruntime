// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <utility>

#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"

#if defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/ep/runtime_compatibility.h"
#endif

namespace onnxruntime {
namespace webgpu {

GpuBufferAllocator::GpuBufferAllocator(
    int context_id,
    std::function<const BufferManager&()> buffer_manager_getter,
    std::function<CommandRecordingState&()> recording_getter,
    bool is_read_only_allocator,
    std::function<bool()> should_submit_zero_initialize,
    std::function<const BufferManager&()> streamless_buffer_manager_getter,
    std::function<void(WGPUBuffer, const CommandRecordingState*)> buffer_releaser)
    : IAllocator(
          OrtMemoryInfo(WEBGPU_BUFFER,
                        is_read_only_allocator ? OrtAllocatorType::OrtReadOnlyAllocator
                                               : OrtAllocatorType::OrtDeviceAllocator,
                        WebGpuDevice(context_id),
                        OrtMemTypeDefault)),
      buffer_manager_getter_{std::move(buffer_manager_getter)},
      streamless_buffer_manager_getter_{streamless_buffer_manager_getter ? std::move(streamless_buffer_manager_getter)
                                                                         : buffer_manager_getter_},
      recording_getter_{std::move(recording_getter)},
      should_submit_zero_initialize_{std::move(should_submit_zero_initialize)},
      buffer_releaser_{std::move(buffer_releaser)},
      mapped_at_creation_{is_read_only_allocator && buffer_manager_getter_().SupportsUMA()},
      initialize_to_zero_{!is_read_only_allocator} {
}

// Streamless allocation, e.g., application CreateTensor/Alloc APIs using a Session allocator,
// or framework allocations without a stream, including during Run. Serialized-mode Session allocators
// defer clears during Run, where streamless copies flush the active Session before using another recording.
// Other plugin allocations submit independent clears. Built-in callers can supply a different policy.
void* GpuBufferAllocator::Alloc(size_t size) {
#if defined(ORT_USE_EP_API_ADAPTERS)
  if (ep::UseSerializedExecutionMode() && recording_getter_ && should_submit_zero_initialize_) {
    auto& recording = recording_getter_();
    std::lock_guard<std::recursive_mutex> lock{recording.mutex};
    return Allocate(size, buffer_manager_getter_(), recording, should_submit_zero_initialize_());
  }
  // Streamless clears are independent of Run and must not read its capture state or per-graph managers.
  CommandRecordingState recording;
  return Allocate(size, streamless_buffer_manager_getter_(), recording, true);
#else
  auto& recording = recording_getter_();
  std::lock_guard<std::recursive_mutex> lock{recording.mutex};
  return Allocate(size, buffer_manager_getter_(), recording,
                  should_submit_zero_initialize_ && should_submit_zero_initialize_());
#endif
}

void* GpuBufferAllocator::Allocate(size_t size, const BufferManager& buffer_manager,
                                   CommandRecordingState& recording, bool submit_zero_initialize) {
  std::lock_guard<std::recursive_mutex> lock{recording.mutex};
  if (size == 0) {
    return nullptr;
  }

  wgpu::BufferUsage usage = mapped_at_creation_ ? wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapWrite
                                                : wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;

  auto buffer = buffer_manager.Create(recording, size, usage, initialize_to_zero_,
                                      submit_zero_initialize);
  num_allocs_.fetch_add(1, std::memory_order_relaxed);
  return buffer;
}

void GpuBufferAllocator::Free(void* p) {
  if (p != nullptr) {
    const auto buffer = static_cast<WGPUBuffer>(p);
    const CommandRecordingState* recording = recording_getter_ ? &recording_getter_() : nullptr;
    if (buffer_releaser_) {
      buffer_releaser_(buffer, recording);
    } else {
      streamless_buffer_manager_getter_().Release(buffer, recording);
    }
    num_allocs_.fetch_sub(1, std::memory_order_relaxed);
  }
}

void GpuBufferAllocator::GetStats(AllocatorStats* stats) {
  *stats = AllocatorStats{};
  stats->num_allocs = num_allocs_.load(std::memory_order_relaxed);
}

WebGpuNoOpAllocator::WebGpuNoOpAllocator(int context_id, bool is_read_only_allocator)
    : IAllocator(
          OrtMemoryInfo(WEBGPU_BUFFER,
                        is_read_only_allocator ? OrtAllocatorType::OrtReadOnlyAllocator
                                               : OrtAllocatorType::OrtDeviceAllocator,
                        WebGpuDevice(context_id),
                        OrtMemTypeDefault)) {
}

void* WebGpuNoOpAllocator::Alloc(size_t /*size*/) {
  ORT_THROW("WebGPU EP device-free context must not allocate device memory.");
}

void WebGpuNoOpAllocator::Free(void* /*p*/) {
}

AllocatorPtr CreateWebGpuAllocator(int context_id,
                                   bool device_free,
                                   std::function<const BufferManager&()> buffer_manager_getter,
                                   std::function<CommandRecordingState&()> recording_getter,
                                   bool is_read_only_allocator,
                                   std::function<bool()> should_submit_zero_initialize,
                                   std::function<const BufferManager&()> streamless_buffer_manager_getter,
                                   std::function<void(WGPUBuffer, const CommandRecordingState*)> buffer_releaser) {
  if (device_free) {
    return std::make_shared<WebGpuNoOpAllocator>(context_id, is_read_only_allocator);
  }
  return std::make_shared<GpuBufferAllocator>(context_id, std::move(buffer_manager_getter), std::move(recording_getter),
                                              is_read_only_allocator,
                                              std::move(should_submit_zero_initialize),
                                              std::move(streamless_buffer_manager_getter),
                                              std::move(buffer_releaser));
}

}  // namespace webgpu
}  // namespace onnxruntime
