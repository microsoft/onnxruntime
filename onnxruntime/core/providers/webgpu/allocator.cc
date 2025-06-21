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

  stats_.num_allocs++;

#if !defined(__wasm__)
  if (!session_initialized_ && context_.DeviceHasFeature(wgpu::FeatureName::BufferMapExtendedUsages)) {
    return context_.BufferManager().CreateUMA(size, session_id_);
  }
#endif  // !defined(__wasm__)

  return context_.BufferManager().Create(size, session_id_);
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

void GpuBufferAllocator::OnSessionInitializationStart(uint32_t session_id) {
  session_id_ = session_id;
}

void GpuBufferAllocator::OnSessionInitializationEnd() {
  session_initialized_ = true;
}

}  // namespace webgpu
}  // namespace onnxruntime
