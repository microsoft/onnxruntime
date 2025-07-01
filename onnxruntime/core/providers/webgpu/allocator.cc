// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

void* GpuBufferAllocator::Reserve(size_t size) {
  ORT_ENFORCE(!session_initialized_, "Session has already been initialized.");

#if !defined(__wasm__)
  if (context_.DeviceHasFeature(wgpu::FeatureName::BufferMapExtendedUsages)) {
    return context_.BufferManager().CreateUMA(size);
  }
#endif  // !defined(__wasm__)

  if (size == 0) {
    return nullptr;
  }

  stats_.num_allocs++;

  return context_.BufferManager().Create(size);
}

void* GpuBufferAllocator::Alloc(size_t size) {
  ORT_ENFORCE(session_initialized_, "Session has not been initialized.");

  if (size == 0) {
    return nullptr;
  }

  stats_.num_allocs++;

  return context_.BufferManager().Create(size);
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
