// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

void* GpuBufferAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  WGPUBuffer buffer;
  if (!session_initialized_ && context_.SupportsBufferMapExtendedUsages()) {
    buffer = context_.BufferManager().CreateUMA(size);
  } else {
    buffer = context_.BufferManager().Create(size);
  }

  stats_.num_allocs++;
  return buffer;
}

void GpuBufferAllocator::Free(void* p) {
  if (p != nullptr) {
    context_.BufferManager().Release(static_cast<WGPUBuffer>(p));
    stats_.num_allocs--;
  }
}

void GpuBufferAllocator::GetStats(AllocatorStats* stats) {
  *stats = stats_;
}

void GpuBufferAllocator::OnSessionInitializationEnd() {
  session_initialized_ = true;
}

}  // namespace webgpu
}  // namespace onnxruntime
