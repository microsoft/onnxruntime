// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_allocator_plugin.h"

namespace onnxruntime {
namespace cuda_plugin {

namespace {

void RestoreDeviceIfKnown(bool restore_prev_device, int prev_device) noexcept {
  if (restore_prev_device) {
    static_cast<void>(cudaSetDevice(prev_device));
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// CudaDeviceAllocator — uses cudaMalloc/cudaFree for GPU device memory.
//
// PERFORMANCE NOTE (Direct cudaMalloc Penalty):
// No arena or caching layer is provided within this plugin. Every allocation
// goes directly to CUDA (cudaMalloc). For models with dynamic shape resizing
// or many intermediate buffers, this can cause substantial overhead.
// Compared to the built-in CUDA Execution Provider, which has an integrated
// memory arena, this is a notable performance gap unless an external
// memory pool/arena is injected or configured by the application.
// ---------------------------------------------------------------------------

CudaDeviceAllocator::CudaDeviceAllocator(const OrtMemoryInfo* memory_info, int device_id)
    : CudaAllocatorBase(CudaAllocatorKind::kDevice, memory_info),
      device_id_(device_id) {
  version = ORT_API_VERSION;
  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = ReserveImpl;
  GetStats = nullptr;
  AllocOnStream = nullptr;
}

/*static*/ void* ORT_API_CALL CudaDeviceAllocator::AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept {
  auto* alloc = static_cast<CudaDeviceAllocator*>(this_ptr);
  void* p = nullptr;
  if (size == 0) return nullptr;
  // Save and restore CUDA device context to avoid corrupting the calling
  // thread's device state in multi-GPU scenarios.
  int prev_device = -1;
  const bool restore_prev_device = cudaGetDevice(&prev_device) == cudaSuccess;
  if (cudaSetDevice(alloc->device_id_) != cudaSuccess) {
    RestoreDeviceIfKnown(restore_prev_device, prev_device);
    return nullptr;
  }
  cudaError_t err = cudaMalloc(&p, size);
  RestoreDeviceIfKnown(restore_prev_device, prev_device);
  if (err != cudaSuccess) {
    return nullptr;
  }
  return p;
}

/*static*/ void ORT_API_CALL CudaDeviceAllocator::FreeImpl(OrtAllocator* this_ptr, void* p) noexcept {
  auto* alloc = static_cast<CudaDeviceAllocator*>(this_ptr);
  if (p != nullptr) {
    int prev_device = -1;
    const bool restore_prev_device = cudaGetDevice(&prev_device) == cudaSuccess;
    if (cudaSetDevice(alloc->device_id_) != cudaSuccess) {
      RestoreDeviceIfKnown(restore_prev_device, prev_device);
      return;
    }

    static_cast<void>(cudaFree(p));
    RestoreDeviceIfKnown(restore_prev_device, prev_device);
  }
}

/*static*/ const OrtMemoryInfo* ORT_API_CALL CudaDeviceAllocator::InfoImpl(const OrtAllocator* this_ptr) noexcept {
  const auto* alloc = static_cast<const CudaDeviceAllocator*>(this_ptr);
  return alloc->GetMemoryInfo();
}

/*static*/ void* ORT_API_CALL CudaDeviceAllocator::ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept {
  // Reserve currently delegates to Alloc (no separate reservation pool).
  return AllocImpl(this_ptr, size);
}

// ---------------------------------------------------------------------------
// CudaPinnedAllocator — uses cudaHostAlloc/cudaFreeHost for page-locked
// host memory visible to the GPU.
// ---------------------------------------------------------------------------

CudaPinnedAllocator::CudaPinnedAllocator(const OrtMemoryInfo* memory_info)
    : CudaAllocatorBase(CudaAllocatorKind::kPinned, memory_info) {
  version = ORT_API_VERSION;
  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = ReserveImpl;
  GetStats = nullptr;
  AllocOnStream = nullptr;
}

/*static*/ void* ORT_API_CALL CudaPinnedAllocator::AllocImpl(OrtAllocator* /*this_ptr*/, size_t size) noexcept {
  void* p = nullptr;
  if (size == 0) return nullptr;
  cudaError_t err = cudaHostAlloc(&p, size, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    return nullptr;
  }
  return p;
}

/*static*/ void ORT_API_CALL CudaPinnedAllocator::FreeImpl(OrtAllocator* /*this_ptr*/, void* p) noexcept {
  if (p != nullptr) {
    cudaFreeHost(p);
  }
}

/*static*/ const OrtMemoryInfo* ORT_API_CALL CudaPinnedAllocator::InfoImpl(const OrtAllocator* this_ptr) noexcept {
  const auto* alloc = static_cast<const CudaPinnedAllocator*>(this_ptr);
  return alloc->GetMemoryInfo();
}

/*static*/ void* ORT_API_CALL CudaPinnedAllocator::ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept {
  // Reserve currently delegates to Alloc (no separate reservation pool).
  return AllocImpl(this_ptr, size);
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
