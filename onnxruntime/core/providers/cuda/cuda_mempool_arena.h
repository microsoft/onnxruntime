// Copyright (c) Microsoft.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>
#include <mutex>

#include "core/common/common.h"                      // ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE, ORT_THROW/ENFORCE
#include "core/common/inlined_containers.h"          // InlinedHashMap, InlinedHashSet, InlinedVector
#include "core/framework/allocator.h"                // IAllocator, IArena (work branch), Stream, OrtMemoryInfo, AllocatorStats
#include "core/providers/cuda/cuda_stream_handle.h"  // ORT Stream -> cudaStream_t

namespace onnxruntime {
namespace cuda {

/**
 * @brief Stream-aware CUDA allocator implemented on top of a private `cudaMemPool_t`.
 *
 * ### Behavior
 * - Creates a **process-local** CUDA mempool for a specific device (from `OrtMemoryInfo`).
 * - All allocations use **`cudaMallocFromPoolAsync()`** on either the legacy default stream (0) or a
 *   caller-provided stream. The allocation stream is recorded for ordered free.
 * - `Free()` and `ReleaseStreamBuffers()` enqueue **`cudaFreeAsync()`** on the recorded stream to
 *   respect CUDA's stream-ordered semantics.
 * - `Shrink()` trims the pool with **`cudaMemPoolTrimTo(bytes_to_keep)`** and right-sizes the book-keeping maps
 *   under lock.
 *
 * ### Tuning
 * - `pool_release_threshold`: if non-zero, sets `cudaMemPoolAttrReleaseThreshold`. **Recommended: 1 MB.**
 * - `initial_pool_size_bytes`: if > 0, pre‑reserve pool capacity by setting
 *   `cudaMemPoolAttrReservedMemCurrent`. **Recommended: 10 MB.**
 *
 * ### Thread-safety
 * - All updates to internal maps and statistics are guarded by an internal `std::mutex`.
 *
 * @note The allocator **does not** set the device default mempool and **does not** switch the current device.
 */
class CudaMempoolArena final : public IArena {
 public:
  /**
   * @brief Construct a `CudaMempoolArena` with a private CUDA mempool.
   *
   * @param memory_info              `OrtMemoryInfo` whose device id selects the CUDA device.
   * @param pool_release_threshold   Optional release threshold (bytes) for `cudaMemPoolAttrReleaseThreshold`.
   *                                 If 0, the attribute is not set. **Recommended value: 1 MB.**
   * @param bytes_to_keep            Target size retained by `Shrink()` (passed to `cudaMemPoolTrimTo`).
   * @param initial_pool_size_bytes  If > 0, pre‑reserve pool capacity by setting
   *                                 `cudaMemPoolAttrReservedMemCurrent`. **Recommended value: 10 MB.**
   *
   * The created pool is process-local and is **not** set as the device default pool.
   */
  CudaMempoolArena(const OrtMemoryInfo& memory_info,
                   uint64_t pool_release_threshold,
                   size_t bytes_to_keep,
                   size_t initial_pool_size_bytes);

  /**
   * @brief Destructor:
   *  1) Enqueues cudaFreeAsync() for any outstanding allocations.
   *  2) Synchronizes all known streams (best-effort; ignores invalid handles).
   *  3) Calls cudaDeviceSynchronize() as a final barrier to ensure queued frees complete.
   *  4) Trims pool to zero and destroys it.
   */
  ~CudaMempoolArena() override;

  // -------- IAllocator overrides --------

  /**
   * @brief Allocate @p size bytes using the legacy default CUDA stream (0).
   * @return device pointer or nullptr when size == 0
   * @throws on allocation failure
   */
  void* Alloc(size_t size) override;

  /**
   * @brief Allocate @p size bytes on the given ORT stream (uses `cudaMallocFromPoolAsync`).
   * @return device pointer or nullptr when size == 0
   * @throws on allocation failure
   */
  void* AllocOnStream(size_t size, Stream* stream) override;

  /**
   * @brief Enqueue an ordered async free on the stream that allocated @p p.
   * No-op if @p p is null or not owned by this allocator.
   */
  void Free(void* p) override;

  /**
   * @brief Reserve @p size bytes; implemented in terms of `Alloc(size)`.
   * @return device pointer or nullptr when size == 0
   * @throws on allocation failure
   */
  void* Reserve(size_t size) override { return Alloc(size); }

  /// @brief This allocator is stream-aware.
  bool IsStreamAware() const override { return true; }

  /// @brief Populate basic allocation statistics.
  void GetStats(AllocatorStats* stats) override;

  // -------- IArena overrides --------

  /**
   * @brief Enqueue `cudaFreeAsync()` for all allocations made on @p stream.
   * This method does not synchronize; it only schedules device frees on that stream.
   */
  void ReleaseStreamBuffers(Stream* stream) override;

  /**
   * @brief Trim the pool to `bytes_to_keep` (configured at construction) using `cudaMemPoolTrimTo()`.
   * Also rehashes internal maps under lock to keep them reasonably sized.
   */
  Status Shrink() override;

 public:
  // Make deletion of copy/move operations visible at public scope.
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaMempoolArena);

 private:
  /// Convert ORT `Stream*` to native `cudaStream_t`; null means legacy default (0).
  static cudaStream_t ResolveCudaStream(Stream* stream) noexcept;

  /// Rehash internal maps under lock; invoked only by `Shrink()`.
  void MaybeRehashLocked();

  /// Best-effort synchronization of all streams in stream_map_. Non-throwing; ignores errors.
  void SyncAllKnownStreams_NoThrow();

  struct AllocationRecord {
    size_t bytes;
    cudaStream_t stream;  // stream on which allocation/free are ordered
  };

  // ---- Pool/context configuration (immutable) ----
  const int device_id_;
  uint64_t pool_release_threshold_;
  size_t bytes_to_keep_;
  size_t initial_pool_size_bytes_;
  cudaMemPool_t pool_{nullptr};

  // ---- Bookkeeping (guarded by mutex_) ----
  std::mutex mutex_;
  InlinedHashMap<void*, AllocationRecord> alloc_map_;               // ptr -> record
  InlinedHashMap<cudaStream_t, InlinedHashSet<void*>> stream_map_;  // stream -> ptrs

  // ---- Stats (guarded by mutex_) ----
  size_t total_allocated_ = 0;
  size_t in_use_bytes_ = 0;
  size_t max_bytes_in_use_ = 0;
  size_t num_allocs_ = 0;
  size_t num_arena_shrinkages_ = 0;
  size_t max_alloc_size_ = 0;
};

}  // namespace cuda
}  // namespace onnxruntime
