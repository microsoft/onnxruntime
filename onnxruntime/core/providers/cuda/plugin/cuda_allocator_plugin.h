// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"

namespace onnxruntime {
namespace cuda_plugin {

/// CUDA device memory allocator using cudaMalloc/cudaFree.
class CudaDeviceAllocator : public OrtAllocator {
 public:
  CudaDeviceAllocator(const OrtMemoryInfo* memory_info, int device_id);
  ~CudaDeviceAllocator() = default;

 private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept;
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_ptr, void* p) noexcept;
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtAllocator* this_ptr) noexcept;
  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept;

  const OrtMemoryInfo* memory_info_;
  int device_id_;
};

/// CUDA pinned (host) memory allocator using cudaHostAlloc/cudaFreeHost.
class CudaPinnedAllocator : public OrtAllocator {
 public:
  CudaPinnedAllocator(const OrtMemoryInfo* memory_info);
  ~CudaPinnedAllocator() = default;

 private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_ptr, size_t size) noexcept;
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_ptr, void* p) noexcept;
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_ptr) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtAllocator* this_ptr) noexcept;
  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_ptr, size_t size) noexcept;

  const OrtMemoryInfo* memory_info_;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
