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
  GpuBufferAllocator(const BufferManager& buffer_manager, bool is_read_only_allocator);
  GpuBufferAllocator(std::function<const BufferManager&()> buffer_manager_getter, bool is_read_only_allocator);

  // Re-reads the buffer manager from the getter and caches it.
  // No-op if constructed with a direct BufferManager reference (no getter).
  void RefreshBufferManager();

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

 private:
  AllocatorStats stats_;
  std::function<const BufferManager&()> buffer_manager_getter_;  // may be empty
  // Cached pointer to the active BufferManager, updated by RefreshBufferManager().
  // Lifetime guarantee: the referenced BufferManager is either the context's default
  // (which outlives the EP) or a per-graph buffer manager owned by the EP's
  // per_graph_buffer_mgrs_ map. Both outlive this allocator since the EP owns it.
  // RefreshBufferManager() must be called when the active buffer manager changes
  // (e.g., in OnRunStart/OnRunEnd).
  const BufferManager* buffer_manager_;
  bool mapped_at_creation_;
};

}  // namespace webgpu
}  // namespace onnxruntime
