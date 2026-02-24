// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_allocator_plugin.h"

namespace onnxruntime {
namespace cuda_plugin {

// ---------------------------------------------------------------------------
// CudaDeviceAllocator
// ---------------------------------------------------------------------------

CudaDeviceAllocator::CudaDeviceAllocator(const OrtMemoryInfo* memory_info, int device_id)
    : OrtAllocator{},
      memory_info_(memory_info),
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
  cudaSetDevice(alloc->device_id_);
  cudaError_t err = cudaMalloc(&p, size);
  if (err != cudaSuccess) {
    return nullptr;
  }
  return p;
}

/*static*/ void ORT_API_CALL CudaDeviceAllocator::FreeImpl(OrtAllocator* this_ptr, void* p) noexcept {
  auto* alloc = static_cast<CudaDeviceAllocator*>(this_ptr);
  if (p != nullptr) {
    cudaSetDevice(alloc->device_id_);
    cudaFree(p);
  }
}

/*static*/ const OrtMemoryInfo* ORT_API_CALL CudaDeviceAllocator::InfoImpl(const OrtAllocator* this_ptr) noexcept {
  const auto* alloc = static_cast<const CudaDeviceAllocator*>(this_ptr);
  return alloc->memory_info_;
}

/*static*/ void ORT_API_CALL CudaDeviceAllocator::ReleaseImpl(OrtAllocator* this_ptr) noexcept {
  delete static_cast<CudaDeviceAllocator*>(this_ptr);
}

/*static*/ void* ORT_API_CALL CudaDeviceAllocator::ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept {
  // Reserve uses the same allocation as Alloc for now (no arena)
  return AllocImpl(this_ptr, size);
}

// ---------------------------------------------------------------------------
// CudaPinnedAllocator
// ---------------------------------------------------------------------------

CudaPinnedAllocator::CudaPinnedAllocator(const OrtMemoryInfo* memory_info)
    : OrtAllocator{},
      memory_info_(memory_info) {
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
  return alloc->memory_info_;
}

/*static*/ void ORT_API_CALL CudaPinnedAllocator::ReleaseImpl(OrtAllocator* this_ptr) noexcept {
  delete static_cast<CudaPinnedAllocator*>(this_ptr);
}

/*static*/ void* ORT_API_CALL CudaPinnedAllocator::ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept {
  return AllocImpl(this_ptr, size);
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
