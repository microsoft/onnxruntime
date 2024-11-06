// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "core/framework/session_state.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

void* GpuBufferAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  auto buffer = context_.BufferManager().Create(size);

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

}  // namespace webgpu
}  // namespace onnxruntime
