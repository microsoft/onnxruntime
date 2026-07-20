// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CudaMempoolOrtAllocator: OrtAllocator wrapper around CUDA native memory pools
// (cudaMallocFromPoolAsync / cudaFreeAsync) for the plugin EP.
// Stream-aware, using a process-local cudaMemPool_t per device.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <mutex>

#include "cuda_allocator_plugin.h"
#include "cuda_plugin_utils.h"

#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace cuda_plugin {

/// OrtAllocator wrapper around a private CUDA mempool for stream-ordered allocation.
/// Inherits from CudaAllocatorBase so the factory's ReleaseAllocatorImpl can identify
/// and manage it via GetKind() and pointer-identity matching.
class CudaMempoolOrtAllocator final : public CudaAllocatorBase {
 public:
  /// Config keys recognized in the allocator_options OrtKeyValuePairs.
  struct ConfigKeyNames {
    static constexpr const char* UseCudaMempool = "arena.use_cuda_mempool";
    static constexpr const char* PoolReleaseThreshold = "arena.cuda_mempool_release_threshold";
    static constexpr const char* BytesToKeepOnShrink = "arena.cuda_mempool_bytes_to_keep_on_shrink";
  };

  /// Create a CudaMempoolOrtAllocator for the given memory_info device.
  /// @param memory_info   OrtMemoryInfo identifying the CUDA device.
  /// @param options        Optional config (release threshold, shrink target).
  /// @param api            The OrtApi for logging and KVP operations.
  /// @param logger         The OrtLogger for diagnostic messages.
  /// @param[out] out       Receives the created allocator on success.
  /// @return nullptr on success, OrtStatus* on failure.
  static OrtStatus* Create(const OrtMemoryInfo* memory_info,
                           const OrtKeyValuePairs* options,
                           const OrtApi& api,
                           const OrtLogger& logger,
                           std::unique_ptr<CudaMempoolOrtAllocator>& out);

  ~CudaMempoolOrtAllocator();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaMempoolOrtAllocator);

 private:
  CudaMempoolOrtAllocator(const OrtMemoryInfo* memory_info,
                          const OrtApi& api,
                          const OrtLogger& logger,
                          cudaMemPool_t pool,
                          int device_id,
                          uint64_t pool_release_threshold,
                          size_t bytes_to_keep_on_shrink);

  // OrtAllocator callback implementations
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_, size_t size) noexcept;
  static void* ORT_API_CALL AllocOnStreamImpl(OrtAllocator* this_, size_t size,
                                              OrtSyncStream* stream) noexcept;
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_, void* p) noexcept;
  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_, size_t size) noexcept;
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_) noexcept;
  static OrtStatus* ORT_API_CALL GetStatsImpl(const OrtAllocator* this_,
                                              OrtKeyValuePairs** out) noexcept;
  static OrtStatus* ORT_API_CALL ShrinkImpl(OrtAllocator* this_) noexcept;

  /// Allocate size bytes on the given CUDA stream.
  void* AllocInternal(size_t size, cudaStream_t stream);

  /// Resolve OrtSyncStream* to cudaStream_t; null → legacy default stream (0).
  cudaStream_t ResolveCudaStream(OrtSyncStream* stream) const;

  /// Best-effort synchronization of all streams that have live allocations.
  void SyncAllKnownStreams() noexcept;

  struct AllocationRecord {
    size_t bytes;
    cudaStream_t stream;
  };

  const OrtApi& ort_api_;
  const OrtLogger& logger_;
  int device_id_{0};  // CUDA ordinal for cudaSetDevice guards

  cudaMemPool_t pool_{nullptr};
  uint64_t pool_release_threshold_;
  size_t bytes_to_keep_on_shrink_;

  // Bookkeeping (guarded by mutex_)
  mutable std::mutex mutex_;
  InlinedHashMap<void*, AllocationRecord> alloc_map_;
  InlinedHashMap<cudaStream_t, InlinedHashSet<void*>> stream_map_;

  // Stats (guarded by mutex_)
  size_t in_use_bytes_ = 0;
  size_t max_bytes_in_use_ = 0;
  size_t num_allocs_ = 0;
  size_t max_alloc_size_ = 0;
  size_t num_arena_shrinkages_ = 0;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
