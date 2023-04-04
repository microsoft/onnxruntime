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
#include "core/providers/cuda/cuda_stream_handle.h"

namespace onnxruntime {
namespace cuda {
namespace test {
// TODO: Since the "DeferredRelease" has been migrated to CudaStream class,
// we should migrate this test from CudaEP unit test to CudaStream unit test.
bool TestDeferredRelease() {
  // Create CUDA EP.
  CUDAExecutionProviderInfo info;
  CUDAExecutionProvider ep(info);
  // Initialize allocators in EP.
  onnxruntime::AllocatorManager allocator_manager;
  ep.RegisterAllocator(allocator_manager);
  AllocatorPtr gpu_alloctor = ep.GetAllocator(OrtMemType::OrtMemTypeDefault);
  // Allocator for call cudaMallocHost and cudaFreeHost
  // For details, see CUDAPinnedAllocator in cuda_allocator.cc.
  AllocatorPtr cpu_pinned_alloc = ep.GetAllocator(OrtMemTypeCPU);
  // let the CudaStream instance "own" the default stream, so we can avoid the
  // work to initialize cublas/cudnn/... It is ok since it is just a customized unit test.
  CudaStream stream(nullptr, gpu_alloctor->Info().device, cpu_pinned_alloc, false, true, nullptr, nullptr);
  // 10 MB
  const size_t n_bytes = 10 * 1000000;
  const int64_t n_allocs = 64;
  ORT_THROW_IF_ERROR(ep.OnRunStart());
  for (size_t i = 0; i < n_allocs; ++i) {
    // Allocate 10MB CUDA pinned memory.
    auto pinned_buffer = ep.AllocateBufferOnCPUPinned<void>(n_bytes);
    // Release it using CUDA callback.
    stream.EnqueDeferredCPUBuffer(pinned_buffer.release());
  }

  // Memory stats
  AllocatorStats stats;
  cpu_pinned_alloc->GetStats(&stats);
  ORT_ENFORCE(stats.num_allocs == n_allocs);
  ORT_THROW_IF_ERROR(stream.CleanUpOnRunEnd());
  ORT_THROW_IF_ERROR(ep.OnRunEnd(true));
  return true;
}

bool TestDeferredReleaseWithoutArena() {
  // Create CUDA EP.
  CUDAExecutionProviderInfo info;
  CUDAExecutionProvider ep(info);
  // Initialize allocators in EP.
  onnxruntime::AllocatorManager allocator_manager;

  OrtDevice pinned_device{OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, DEFAULT_CPU_ALLOCATOR_DEVICE_ID};
  // Create allocator without BFCArena
  AllocatorCreationInfo pinned_memory_info(
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<CUDAPinnedAllocator>(device_id, CUDA_PINNED);
      },
      pinned_device.Id(),
      false /* no arena */);
  auto cuda_pinned_alloc = CreateAllocator(pinned_memory_info);
  allocator_manager.InsertAllocator(cuda_pinned_alloc);
  // Use existing allocator in allocator_manager.
  // Also register new allocator created by this EP in allocator_manager.
  ep.RegisterAllocator(allocator_manager);
  AllocatorPtr gpu_alloctor = ep.GetAllocator(OrtMemType::OrtMemTypeDefault);
  // Allocator for call cudaMallocHost and cudaFreeHost
  // For details, see CUDAPinnedAllocator in cuda_allocator.cc.
  AllocatorPtr cpu_pinned_alloc = ep.GetAllocator(OrtMemTypeCPU);
  // let the CudaStream instance "own" the default stream, so we can avoid the
  // work to initialize cublas/cudnn/... It is ok since it is just a customized unit test.
  CudaStream stream(nullptr, gpu_alloctor->Info().device, cpu_pinned_alloc, false, true, nullptr, nullptr);
  // 10 MB
  const size_t n_bytes = 10 * 1000000;
  const int64_t n_allocs = 64;
  ORT_THROW_IF_ERROR(ep.OnRunStart());
  for (size_t i = 0; i < n_allocs; ++i) {
    // Allocate 10MB CUDA pinned memory.
    auto pinned_buffer = ep.AllocateBufferOnCPUPinned<void>(n_bytes);
    // Release it using CUDA callback.
    stream.EnqueDeferredCPUBuffer(pinned_buffer.release());
  }

  ORT_THROW_IF_ERROR(stream.CleanUpOnRunEnd());
  ORT_THROW_IF_ERROR(ep.OnRunEnd(true));
  return true;
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
#endif
