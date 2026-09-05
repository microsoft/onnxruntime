// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <mutex>

#include "core/framework/allocator.h"
#include "core/framework/ortdevice.h"

namespace onnxruntime {
namespace webgpu {

class BufferManager;
struct CommandRecordingState;
class WebGpuContext;

inline constexpr OrtDevice WebGpuDevice{OrtDevice::GPU,
                                        OrtDevice::MemType::DEFAULT,
                                        OrtDevice::VendorIds::NONE,
                                        0};

class GpuBufferAllocator : public IAllocator {
 public:
  // Calls buffer_manager_getter on every Alloc/Free to obtain the current
  // BufferManager. This allows the EP to route allocations to different
  // buffer managers (e.g., per-graph) without explicit refresh calls.
  GpuBufferAllocator(std::function<const BufferManager&()> buffer_manager_getter,
                     std::function<CommandRecordingState&()> recording_getter,
                     bool is_read_only_allocator,
                     std::function<bool()> should_submit_zero_initialize = {});

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

 private:
  AllocatorStats stats_;
  std::function<const BufferManager&()> buffer_manager_getter_;
  std::function<CommandRecordingState&()> recording_getter_;
  std::function<bool()> should_submit_zero_initialize_;
  bool mapped_at_creation_;
  // Cached writable buffers are cleared explicitly by BufferManager::Create. Fresh buffers rely on Dawn's
  // "lazy_clear_resource_on_first_use" toggle, which is enabled by WebGpuContext.
  bool initialize_to_zero_;
};

// Environment-level shared allocator. External tensors are not transient session workspace, so
// they bypass the context BufferManager and do not participate in a session recording timeline.
class ExternalGpuBufferAllocator : public IAllocator {
 public:
  explicit ExternalGpuBufferAllocator(std::shared_ptr<WebGpuContext> context);

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

 private:
  std::shared_ptr<WebGpuContext> context_;
  std::mutex stats_mutex_;
  AllocatorStats stats_;
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
                                   std::function<CommandRecordingState&()> recording_getter,
                                   bool is_read_only_allocator,
                                   std::function<bool()> should_submit_zero_initialize = {});

}  // namespace webgpu
}  // namespace onnxruntime
