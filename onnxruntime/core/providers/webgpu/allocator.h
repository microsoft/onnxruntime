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

}  // namespace webgpu
}  // namespace onnxruntime
