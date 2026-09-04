// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <utility>

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
      is_read_only_allocator_{is_read_only_allocator},
      initialize_to_zero_{!is_read_only_allocator} {
}

void* GpuBufferAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  stats_.num_allocs++;

  const auto& buffer_manager = buffer_manager_getter_();
  if (!mapped_at_creation_.has_value()) {
    mapped_at_creation_ = is_read_only_allocator_ && buffer_manager.SupportsUMA();
  }
  wgpu::BufferUsage usage = *mapped_at_creation_ ? wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapWrite
                                                : wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;

  const bool submit_zero_initialize = should_submit_zero_initialize_ && should_submit_zero_initialize_();
  return buffer_manager.Create(size, usage, initialize_to_zero_, submit_zero_initialize);
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
