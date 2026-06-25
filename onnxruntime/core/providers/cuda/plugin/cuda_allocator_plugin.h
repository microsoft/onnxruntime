// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CUDA device and pinned memory allocator implementations for the plugin EP.
// Provides CudaDeviceAllocator (cudaMalloc/cudaFree) and CudaPinnedAllocator
// (cudaHostAlloc/cudaFreeHost) conforming to the OrtAllocator interface.
// No arena or caching layer; every allocation goes directly to CUDA.

#pragma once

#include "cuda_plugin_utils.h"

namespace onnxruntime {
namespace cuda_plugin {

/// Allocator type: device memory (GPU) or pinned (page-locked host) memory.
enum class CudaAllocatorKind {
  kDevice,  ///< GPU device memory via cudaMalloc
  kPinned,  ///< Page-locked host memory via cudaHostAlloc
};

/// Base class for CUDA allocators implementing the OrtAllocator C interface.
class CudaAllocatorBase : public OrtAllocator {
 public:
  explicit CudaAllocatorBase(CudaAllocatorKind kind, const OrtMemoryInfo* memory_info)
      : OrtAllocator{},
        kind_(kind),
        memory_info_(memory_info) {}

  CudaAllocatorKind GetKind() const { return kind_; }
  const OrtMemoryInfo* GetMemoryInfo() const { return memory_info_; }

 private:
  CudaAllocatorKind kind_;
  const OrtMemoryInfo* memory_info_;
};

/// CUDA device memory allocator using cudaMalloc/cudaFree.
/// Lifetime is managed by the EP factory (ReleaseAllocatorImpl), not by a Release callback.
class CudaDeviceAllocator final : public CudaAllocatorBase {
 public:
  CudaDeviceAllocator(const OrtMemoryInfo* memory_info, int device_id);
  ~CudaDeviceAllocator() = default;

 private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept;
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_ptr, void* p) noexcept;
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_ptr) noexcept;
  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept;

  int device_id_;
};

/// CUDA pinned (host) memory allocator using cudaHostAlloc/cudaFreeHost.
/// Lifetime is managed by the EP factory (ReleaseAllocatorImpl), not by a Release callback.
class CudaPinnedAllocator final : public CudaAllocatorBase {
 public:
  CudaPinnedAllocator(const OrtMemoryInfo* memory_info);
  ~CudaPinnedAllocator() = default;

 private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept;
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_ptr, void* p) noexcept;
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_ptr) noexcept;
  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
