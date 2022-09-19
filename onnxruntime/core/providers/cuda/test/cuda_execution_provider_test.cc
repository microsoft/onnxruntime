// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This test is built only under DEBUG mode because it requires
// extra code in the core of CUDA EP and that code may
//  1. slow down performance critical applications and
//  2. increase binary size of ORT.
#ifndef NDEBUG
#include <iostream>
#include "core/providers/cuda/test/all_tests.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda {
namespace test {

bool TestDeferredRelease() {
  OrtArenaCfg ort_arena_cfg;
  // 1 means ArenaExtendStrategy::kSameAsRequested.
  // See allocator.h for details.
  ort_arena_cfg.arena_extend_strategy = 1;
  // Create CUDA EP.
  CUDAExecutionProviderInfo info;
  CUDAExecutionProvider ep(info);
  // Initialize allocators in EP.
  onnxruntime::AllocatorManager allocator_manager;
  ep.RegisterAllocator(allocator_manager);
  // Allocator for call cudaMallocHost and cudaFreeHost
  // For details, see CUDAPinnedAllocator in cuda_allocator.cc.
  AllocatorPtr cpu_pinned_alloc = ep.GetAllocator(DEFAULT_CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPU);
  // 10 MB
  const size_t n_bytes = 10 * 1000000;
  const int64_t n_allocs = 64;
  ORT_THROW_IF_ERROR(ep.OnRunStart());
  for (size_t i = 0; i < n_allocs; ++i) {
    // Allocate 10MB CUDA pinned memory.
    auto pinned_buffer = ep.AllocateBufferOnCPUPinned<void>(n_bytes);
    // Release it using CUDA callback.
    ep.AddDeferredReleaseCPUPtr(pinned_buffer.release());
  }

  // Memory stats
  AllocatorStats stats;
  cpu_pinned_alloc->GetStats(&stats);
  ORT_ENFORCE(stats.num_allocs == n_allocs);
  ORT_THROW_IF_ERROR(ep.OnRunEnd(true));
  return true;
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
#endif
