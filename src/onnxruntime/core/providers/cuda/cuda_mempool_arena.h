// Copyright (c) Microsoft.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>
#include <mutex>

#include "core/common/common.h"                      // ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE, ORT_THROW/ENFORCE
#include "core/common/inlined_containers.h"          // InlinedHashMap, InlinedHashSet, InlinedVector
#include "core/providers/cuda/cuda_stream_handle.h"  // ORT Stream -> cudaStream_t
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace logging {
class Logger;
}
namespace cuda {
/**
 * @brief Stream-aware CUDA allocator implemented on top of a private `cudaMemPool_t`.
 *        The purpose of this arena is to assist with memory allocations in environments where
 *        a single process is hosting more than one cuda session. This arena hosts cuda memory pool
 *        which has some tunable parameters to control its memory usage and de-allocates memory back to
 *        the device according to the specified params. This is opposite to the BFCArena which only
 *        attempts to free memory on Shrink() if configured at the end of the run.
 *
 * ### Behavior
 * - Creates a **process-local** CUDA mempool for a specific device (from `OrtMemoryInfo`).
 * - All allocations use **`cudaMallocFromPoolAsync()`** on either the legacy default stream (0) or a
 *   caller-provided stream. The allocation stream is recorded for ordered free.
 * - `Free()` enqueue **`cudaFreeAsync()`** on the recorded stream to
 *   respect CUDA's stream-ordered semantics.
 * - `Shrink()` trims the pool with **`cudaMemPoolTrimTo(bytes_to_keep)`** and right-sizes the book-keeping maps
 *   under lock.
 *
 * ### Tuning
 * - `pool_release_threshold`: if non-zero, sets `cudaMemPoolAttrReleaseThreshold`. **Recommended: 1 MB.**, but
 *    must be experimentally determined based on workload for optimal memory consumption vs performance.
 *   `cudaMemPoolAttrReservedMemCurrent`. **Recommended: 10 MB.**
 * - `bytes_to_keep_on_shrink`: target size for `cudaMemPoolTrimTo()` on `Shrink()`. This is only relevant
 *    if Shrink() is enabled. It usually costs performance, and strictly speaking is not necessary for cuda mempools
 *    since they release memory on at synchronous points according to `pool_release_threshold`.
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
   * @param bytes_to_keep_on_shrink  Target size (bytes) for `cudaMemPoolTrimTo()` on `Shrink()`.
   * @param logger                   Cuda EP Logger
   *
   * The created pool is process-local and is **not** set as the device default pool.
   */
  CudaMempoolArena(const OrtMemoryInfo& memory_info,
                   uint64_t pool_release_threshold,
                   size_t bytes_to_keep_on_shrink,
                   const logging::Logger* logger);

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
   *   This is done so all the memory is gone including initializers when
   *   the session is torn down.
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
   * we intentionally do not implement this method. The call to this method
   * will yank memory from under live OrtValues such as allocated for output
   * bound and the resulting output OrtValue will not be valid.
   * Then when the OrtValues attempt to release memory those entries are not found
   * in the map: CudaMempoolArena::Free: pointer 0000000203800400 not found in allocation map; ignoring
   * The reason this works with BFCArena is because it does not really release memory.
   */
  // void ReleaseStreamBuffers(Stream* stream) override;

  /**
   * @brief Trim the pool to `bytes_to_keep_on_shrink_` (configured at construction) using `cudaMemPoolTrimTo()`.
   * Memory still allocated is not affected. Shrink() may affect your performance and contrary to BFCArena
   * This allocator does not need Shrink. Cuda mempool is capable of releasing memory automatically
   * according to pool_release_threshold_ set at construction.
   * Also rehashes internal maps under lock to keep them reasonably sized.
   */
  Status Shrink() override;

  static bool IsCudaVersionSupported() noexcept;

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
  uint64_t pool_release_threshold_;
  size_t bytes_to_keep_on_shrink_;
  size_t initial_pool_size_bytes_;
  const logging::Logger* logger_;
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
