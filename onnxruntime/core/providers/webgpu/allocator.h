// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "core/framework/allocator.h"
#include "core/framework/ortdevice.h"

namespace onnxruntime {
namespace webgpu {

class BufferManager;

inline constexpr OrtDevice WebGpuDevice{OrtDevice::GPU,
                                        OrtDevice::MemType::DEFAULT,
                                        OrtDevice::VendorIds::NONE,
                                        0};

class GpuBufferAllocator : public IAllocator {
 public:
  // Calls buffer_manager_getter on every Alloc/Free to obtain the current
  // BufferManager. This allows the EP to route allocations to different
  // buffer managers (e.g., per-graph) without explicit refresh calls.
  GpuBufferAllocator(std::function<const BufferManager&()> buffer_manager_getter, bool is_read_only_allocator);

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

 private:
  AllocatorStats stats_;
  std::function<const BufferManager&()> buffer_manager_getter_;
  bool mapped_at_creation_;
};

// No-op allocator used for the WebGPU device when the context has no Dawn device (a device-free /
// "virtual device" context). A real GpuBufferAllocator cannot be constructed without a device (its ctor
// queries the device via BufferManager::SupportsUMA), and such a context only runs graph transformation
// and never allocates. This exposes the same OrtMemoryInfo so the device's allocator contract is met,
// but Alloc/Free are never expected to be called.
class WebGpuNoOpAllocator : public IAllocator {
 public:
  explicit WebGpuNoOpAllocator(bool is_read_only_allocator);

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

// Creates the WebGPU device allocator: a real GpuBufferAllocator when the context has a device, or a
// no-op WebGpuNoOpAllocator for a device-free context, where a real one can't be constructed and no
// allocation ever happens.
AllocatorPtr CreateWebGpuAllocator(bool device_free,
                                   std::function<const BufferManager&()> buffer_manager_getter,
                                   bool is_read_only_allocator);

}  // namespace webgpu
}  // namespace onnxruntime
