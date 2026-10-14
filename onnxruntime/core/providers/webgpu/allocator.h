// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <functional>

#include "core/common/narrow.h"
#include "core/framework/allocator.h"
#include "core/framework/ortdevice.h"

namespace onnxruntime {
namespace webgpu {

class BufferManager;
struct CommandRecordingState;

inline OrtDevice WebGpuDevice(int context_id) {
  return OrtDevice{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE,
                   narrow<OrtDevice::DeviceId>(context_id)};
}

// Shared allocation implementation for native and plugin builds. Session getters borrow the EP;
// plugin Env allocators have no Session recording. The returned objects
// must remain alive throughout allocator use and tensor frees. BufferManager synchronizes deferred
// releases when plugin Session allocators are used concurrently. Plugin ABI wrappers live in ep/allocator.h.
class GpuBufferAllocator : public IAllocator {
 public:
  // Calls buffer_manager_getter on every Alloc/Free to obtain the current
  // BufferManager. This allows the EP to route allocations to different
  // buffer managers (e.g., per-graph) without explicit refresh calls.
  // Read-only initializers skip cached-buffer clears and can be mapped at creation on UMA.
  // should_submit_zero_initialize is used by built-in WebGPU and serialized-mode Session allocators.
  // Concurrent plugin mode ignores it.
  // TODO: Remove this callback once built-in WebGPU can distinguish external allocators used outside Run
  // from internal allocators used during Run.
  // Serialized-mode Session Alloc uses its recording and submits clears outside Run, deferring them during Run.
  // Other plugin Alloc calls submit independent clears.
  // Plugin Env allocators omit recording_getter, as they never use a Session stream.
  GpuBufferAllocator(int context_id,
                     std::function<const BufferManager&()> buffer_manager_getter,
                     std::function<CommandRecordingState&()> recording_getter,
                     bool is_read_only_allocator,
                     std::function<bool()> should_submit_zero_initialize = {});

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

#if defined(ORT_USE_EP_API_ADAPTERS)
  bool IsStreamAware() const override { return static_cast<bool>(recording_getter_); }
  void* AllocOnStream(size_t size, Stream* stream) override;
#endif

 private:
  void* Allocate(size_t size, CommandRecordingState& recording, bool submit_zero_initialize);
  std::atomic<int64_t> num_allocs_{0};
  std::function<const BufferManager&()> buffer_manager_getter_;
  std::function<CommandRecordingState&()> recording_getter_;
  std::function<bool()> should_submit_zero_initialize_;
  bool mapped_at_creation_;
  // Cached writable buffers are cleared explicitly by BufferManager::Create. Fresh buffers rely on Dawn's
  // "lazy_clear_resource_on_first_use" toggle, which is enabled by WebGpuContext.
  bool initialize_to_zero_;
};

// No-op allocator used for the WebGPU device when the context has no Dawn device (a device-free /
// "virtual device" context). A real GpuBufferAllocator cannot be constructed without a device (its ctor
// queries the device via BufferManager::SupportsUMA), and such a context only runs graph transformation
// and never allocates. This exposes the same OrtMemoryInfo so the device's allocator contract is met,
// but Alloc/Free are never expected to be called.
class WebGpuNoOpAllocator : public IAllocator {
 public:
  WebGpuNoOpAllocator(int context_id, bool is_read_only_allocator);

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

// Creates the WebGPU device allocator: a real GpuBufferAllocator when the context has a device, or a
// no-op WebGpuNoOpAllocator for a device-free context, where a real one can't be constructed and no
// allocation ever happens.
AllocatorPtr CreateWebGpuAllocator(int context_id,
                                   bool device_free,
                                   std::function<const BufferManager&()> buffer_manager_getter,
                                   std::function<CommandRecordingState&()> recording_getter,
                                   bool is_read_only_allocator,
                                   std::function<bool()> should_submit_zero_initialize = {});

}  // namespace webgpu
}  // namespace onnxruntime
