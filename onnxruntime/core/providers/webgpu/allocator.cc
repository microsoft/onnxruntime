// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"

namespace onnxruntime {
namespace webgpu {

void* GpuBufferAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  stats_.num_allocs++;

#if !defined(__wasm__)
  // Check if the buffer manager supports UMA and we're not yet in an initialized session
  if (!session_initialized_ && buffer_manager_.SupportsUMA()) {
    return buffer_manager_.CreateUMA(size);
  }
#endif  // !defined(__wasm__)

  return buffer_manager_.Create(size);
}

void GpuBufferAllocator::Free(void* p) {
  if (p != nullptr) {
    buffer_manager_.Release(static_cast<WGPUBuffer>(p));
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
