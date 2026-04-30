// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"

namespace onnxruntime {
namespace webgpu {

GpuBufferAllocator::GpuBufferAllocator(const BufferManager& buffer_manager, bool is_read_only_allocator)
  : GpuBufferAllocator([buffer_manager_ptr = &buffer_manager]() -> const BufferManager& { return *buffer_manager_ptr; }, is_read_only_allocator) {
}

GpuBufferAllocator::GpuBufferAllocator(std::function<const BufferManager&()> buffer_manager_getter, bool is_read_only_allocator)
    : IAllocator(
          OrtMemoryInfo(WEBGPU_BUFFER,
                        is_read_only_allocator ? OrtAllocatorType::OrtReadOnlyAllocator
                                               : OrtAllocatorType::OrtDeviceAllocator,
                        WebGpuDevice,
                        OrtMemTypeDefault)),
  buffer_manager_getter_{std::move(buffer_manager_getter)},
  mapped_at_creation_{is_read_only_allocator && buffer_manager_getter_().SupportsUMA()} {
}

void* GpuBufferAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  stats_.num_allocs++;

  const auto& buffer_manager = buffer_manager_getter_();

  wgpu::BufferUsage usage = mapped_at_creation_ ? wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapWrite
                                                : wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;

  return buffer_manager.Create(size, usage);
}

void GpuBufferAllocator::Free(void* p) {
  if (p != nullptr) {
    buffer_manager_getter_().Release(static_cast<WGPUBuffer>(p));
    stats_.num_allocs--;
  }
}

void GpuBufferAllocator::GetStats(AllocatorStats* stats) {
  *stats = stats_;
}

}  // namespace webgpu
}  // namespace onnxruntime
