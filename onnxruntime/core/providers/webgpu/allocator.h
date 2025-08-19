// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "core/framework/ortdevice.h"

namespace onnxruntime {
namespace webgpu {

class BufferManager;

class GpuBufferAllocator : public IAllocator {
 public:
  GpuBufferAllocator(const BufferManager& buffer_manager, bool is_read_only_allocator);

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  void GetStats(AllocatorStats* stats) override;

 private:
  AllocatorStats stats_;
  const BufferManager& buffer_manager_;
  bool mapped_at_creation_;
};

}  // namespace webgpu
}  // namespace onnxruntime
