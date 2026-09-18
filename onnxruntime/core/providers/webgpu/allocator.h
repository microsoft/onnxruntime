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

// Plugin device allocators (ORT_USE_EP_API_ADAPTERS, with a real device):
// The Session column describes config.device_allocator, also used for kernel scratch.
// Env APIs can allocate after Env/device setup, before any Session exists, and remain usable afterward.
// Session allocators require an existing Session and serve both application APIs and internal execution.
//
// | Aspect          | Env shared allocator               | Session device allocator                    |
// |-----------------|------------------------------------|---------------------------------------------|
// | App API use     | CreateTensor/Alloc without Session | CreateTensor/Alloc via a Session allocator  |
// | Internal use    | Not used for EP kernel scratch     | Run input/intermediate/output and scratch   |
// | Implementation  | ExternalGpuBufferAllocator         | GpuBufferAllocator                          |
// | Created by      | Factory::CreateAllocatorImpl       | Factory::CreateEpImpl                       |
// | Impl creation   | Lazy, on first allocation          | Once when creating the Session's EP         |
// | C API wrapper   | adapter::Allocator                 | WebGpuSessionAllocator                      |
// | C API exposure  | Factory::CreateAllocatorImpl       | Ep::CreateAllocatorImpl wraps existing impl |
// | Buffer manager  | Context's default BufferManager    | EP-selected context or per-graph manager    |
// | Recording       | Owns command_state_                | Borrows the owning EP's Recording()         |
// | Lifetime        | Retains Context; no Session needed | EP must outlive allocator use/tensor frees  |
// | Alloc           | Submit cached clear before return  | Submit cached clear, even during Run        |
// | AllocOnStream   | Not provided                       | Matching Session stream: defer cached clear |
// |                 |                                    | Null stream: same policy as plain Alloc     |
//
// Either can supply tensors to other Sessions on the same WebGPU device/context. Using a small
// Session only for allocation does not remove its lifetime requirement. A shared buffer cache
// does not imply a shared recording; callers must order tensor writes before another Session uses them.
// Alloc vs AllocOnStream is a stream-based distinction, not an external-vs-internal API distinction:
// BindInput can allocate on a Session stream before Run; streamless allocation during Run still uses Alloc.
// Read-only initializers and writable prepacked weights use separate GpuBufferAllocator instances
// with InitializerBufferManager(), not the device allocator above. Read-only initializers skip clears;
// prepack and native-EP callers can supply different plain-Alloc submission policies.
class GpuBufferAllocator : public IAllocator {
 public:
  // Calls buffer_manager_getter on every Alloc/Free to obtain the current
  // BufferManager. This allows the EP to route allocations to different
  // buffer managers (e.g., per-graph) without explicit refresh calls.
  // Read-only initializers skip cached-buffer clears and can be mapped at creation on UMA.
  // should_submit_zero_initialize controls plain Alloc; a matching plugin AllocOnStream
  // instead defers clears on the supplied Session stream.
  GpuBufferAllocator(std::function<const BufferManager&()> buffer_manager_getter,
                     std::function<CommandRecordingState&()> recording_getter,
                     bool is_read_only_allocator,
                     std::function<bool()> should_submit_zero_initialize = {});

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

#if defined(ORT_USE_EP_API_ADAPTERS)
  bool IsStreamAware() const override { return true; }
  void* AllocOnStream(size_t size, Stream* stream) override;
#endif

 private:
  void* Allocate(size_t size, bool submit_zero_initialize);
  AllocatorStats stats_;
  std::function<const BufferManager&()> buffer_manager_getter_;
  std::function<CommandRecordingState&()> recording_getter_;
  std::function<bool()> should_submit_zero_initialize_;
  bool mapped_at_creation_;
  // Cached writable buffers are cleared explicitly by BufferManager::Create. Fresh buffers rely on Dawn's
  // "lazy_clear_resource_on_first_use" toggle, which is enabled by WebGpuContext.
  bool initialize_to_zero_;
};

class ExternalGpuBufferAllocator : public IAllocator {
 public:
  explicit ExternalGpuBufferAllocator(std::shared_ptr<WebGpuContext> context);
  ~ExternalGpuBufferAllocator() override;

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

 private:
  std::shared_ptr<WebGpuContext> context_;
  std::unique_ptr<CommandRecordingState> command_state_;
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

#if defined(ORT_USE_EP_API_ADAPTERS)
OrtAllocator* CreateWebGpuSessionAllocator(AllocatorPtr allocator);
bool TryReleaseWebGpuSessionAllocator(OrtAllocator* allocator);
#endif

}  // namespace webgpu
}  // namespace onnxruntime
