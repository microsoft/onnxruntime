// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <utility>

#include "core/common/safeint.h"
#include "core/framework/session_state.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"

namespace onnxruntime {
namespace webgpu {

GpuBufferAllocator::GpuBufferAllocator(
    std::function<const BufferManager&()> buffer_manager_getter,
    bool is_read_only_allocator,
    std::function<bool()> should_submit_zero_initialize)
    : IAllocator(
          OrtMemoryInfo(WEBGPU_BUFFER,
                        is_read_only_allocator ? OrtAllocatorType::OrtReadOnlyAllocator
                                               : OrtAllocatorType::OrtDeviceAllocator,
                        WebGpuDevice,
                        OrtMemTypeDefault)),
      buffer_manager_getter_{std::move(buffer_manager_getter)},
      should_submit_zero_initialize_{std::move(should_submit_zero_initialize)},
      mapped_at_creation_{is_read_only_allocator && buffer_manager_getter_().SupportsUMA()},
      initialize_to_zero_{!is_read_only_allocator} {
}

void* GpuBufferAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  wgpu::BufferUsage usage = mapped_at_creation_ ? wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapWrite
                                                : wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;

  const bool submit_zero_initialize = should_submit_zero_initialize_ && should_submit_zero_initialize_();
  void* buffer = buffer_manager_getter_().Create(size, usage, initialize_to_zero_, submit_zero_initialize);
  const size_t allocated_size = static_cast<size_t>(
      wgpuBufferGetSize(static_cast<WGPUBuffer>(buffer)));
  const int64_t allocated_size_int64 = SafeInt<int64_t>(allocated_size);
  const int64_t requested_size_int64 = SafeInt<int64_t>(size);
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    allocations_.emplace(buffer, AllocationSize{size, allocated_size});
    ++stats_.num_allocs;
    stats_.bytes_in_use += allocated_size_int64;
    stats_.bytes_requested_in_use += requested_size_int64;
    stats_.max_bytes_in_use = std::max(stats_.max_bytes_in_use, stats_.bytes_in_use);
    stats_.max_alloc_size =
        std::max(stats_.max_alloc_size, allocated_size_int64);
  }

  return buffer;
}

void GpuBufferAllocator::Free(void* p) {
  if (p != nullptr) {
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      const auto allocation = allocations_.find(p);
      ORT_ENFORCE(allocation != allocations_.end(), "Unknown WebGPU buffer allocation.");
      stats_.bytes_in_use -= SafeInt<int64_t>(allocation->second.allocated);
      stats_.bytes_requested_in_use -= SafeInt<int64_t>(allocation->second.requested);
      allocations_.erase(allocation);
    }
    buffer_manager_getter_().Release(static_cast<WGPUBuffer>(p));
  }
}

void GpuBufferAllocator::GetStats(AllocatorStats* stats) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  *stats = stats_;
}

void GpuBufferAllocator::ResetPeakStats() {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  stats_.max_bytes_in_use = stats_.bytes_in_use;
  stats_.max_alloc_size = 0;
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
                                   bool is_read_only_allocator,
                                   std::function<bool()> should_submit_zero_initialize) {
  if (device_free) {
    return std::make_shared<WebGpuNoOpAllocator>(is_read_only_allocator);
  }
  return std::make_shared<GpuBufferAllocator>(std::move(buffer_manager_getter), is_read_only_allocator,
                                              std::move(should_submit_zero_initialize));
}

}  // namespace webgpu
}  // namespace onnxruntime
