// Copyright (c) Microsoft.
// Licensed under the MIT License.

#include "cuda_mempool_arena.h"

#if ORT_CUDA_HAS_MEMPOOL_API

#include <algorithm>

#include "core/providers/cuda/shared_inc/cuda_call.h"  // ORT CudaCall helpers

namespace onnxruntime {
namespace cuda {

// ======== CudaMempoolArena ========

CudaMempoolArena::CudaMempoolArena(const OrtMemoryInfo& memory_info,
                                   uint64_t pool_release_threshold,
                                   size_t bytes_to_keep,
                                   size_t initial_pool_size_bytes)
    : IArena(memory_info),
      device_id_(memory_info.device.Id()),
      pool_release_threshold_(pool_release_threshold),
      bytes_to_keep_(bytes_to_keep),
      initial_pool_size_bytes_(initial_pool_size_bytes) {
  // Create a process-local device memory pool for device_id_.
  // 'cudaMemAllocationTypeDevice' (for cudaMemPoolProps.allocType) not clear when it is available

  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;    // Pinned is not the same as pinned allocator
  props.handleTypes = cudaMemHandleTypeNone;        // local to process
  props.location.type = cudaMemLocationTypeDevice;  // Device memory
  props.location.id = device_id_;

  CUDA_CALL_THROW(cudaMemPoolCreate(&pool_, &props));

  if (pool_release_threshold_ != 0) {
    CUDA_CALL_THROW(cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold,
                                            &pool_release_threshold_));
  }

  if (initial_pool_size_bytes_ > 0) {
    // Pre-reserve pool backing memory to the requested size.
    size_t reserved = initial_pool_size_bytes_;
    CUDA_CALL_THROW(cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReservedMemCurrent, &reserved));
  }

  // Intentionally DO NOT call cudaDeviceSetMemPool(device_id_, pool_);
  // All allocations explicitly target this pool via cudaMallocFromPoolAsync.
}

CudaMempoolArena::~CudaMempoolArena() {
  // 1) Best-effort: enqueue frees for any remaining allocations on their recorded streams.
  //    No locking by design: destruction implies no concurrent access.
  for (auto& kv : alloc_map_) {
    void* p = kv.first;
    const cudaStream_t s = kv.second.stream;
    (void)cudaFreeAsync(p, s);  // ignore errors in destructor
  }

  // Now it is safe to drop our bookkeeping.
  alloc_map_.clear();
  stream_map_.clear();

  // 2) Synchronize all streams we know about (those that ever held allocations).
  SyncAllKnownStreams_NoThrow();

  // 3) Safety barrier: ensure any frees enqueued on destroyed/unknown streams are completed.
  (void)cudaDeviceSynchronize();  // ignore errors in destructor

  // 4) Trim to zero and destroy the pool.
  if (pool_) {
    (void)cudaMemPoolTrimTo(pool_, 0);  // best-effort
    (void)cudaMemPoolDestroy(pool_);
    pool_ = nullptr;
  }
}

void* CudaMempoolArena::Alloc(size_t size) {
  if (size == 0) return nullptr;

  void* p = nullptr;
  constexpr cudaStream_t kDefaultStream = static_cast<cudaStream_t>(0);
  cudaError_t err = cudaMallocFromPoolAsync(&p, size, pool_, kDefaultStream);
  if (err != cudaSuccess) {
    ORT_THROW("CudaMempoolArena::Alloc: cudaMallocFromPoolAsync failed: ",
              cudaGetErrorString(err), " (", static_cast<int>(err), "), size=", size);
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    AllocationRecord rec{size, kDefaultStream};
    alloc_map_.emplace(p, rec);
    stream_map_[kDefaultStream].insert(p);

    total_allocated_ += size;
    in_use_bytes_ += size;
    max_bytes_in_use_ = std::max(max_bytes_in_use_, in_use_bytes_);
    max_alloc_size_ = std::max(max_alloc_size_, size);
    ++num_allocs_;
  }

  return p;
}

void* CudaMempoolArena::AllocOnStream(size_t size, Stream* stream) {
  if (size == 0) return nullptr;

  void* p = nullptr;
  const cudaStream_t s = ResolveCudaStream(stream);

  cudaError_t err = cudaMallocFromPoolAsync(&p, size, pool_, s);
  if (err != cudaSuccess) {
    ORT_THROW("CudaMempoolArena::AllocOnStream: cudaMallocFromPoolAsync failed on stream=",
              reinterpret_cast<uintptr_t>(s), ": ",
              cudaGetErrorString(err), " (", static_cast<int>(err), "), size=", size);
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    AllocationRecord rec{size, s};
    alloc_map_.emplace(p, rec);
    stream_map_[s].insert(p);

    total_allocated_ += size;
    in_use_bytes_ += size;
    max_bytes_in_use_ = std::max(max_bytes_in_use_, in_use_bytes_);
    max_alloc_size_ = std::max(max_alloc_size_, size);
    ++num_allocs_;
  }

  return p;
}

void CudaMempoolArena::Free(void* p) {
  if (!p) return;

  cudaStream_t s = static_cast<cudaStream_t>(0);
  size_t sz = 0;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = alloc_map_.find(p);
    if (it == alloc_map_.end()) {
      // Not owned by this allocator; ignore per ORT convention.
      return;
    }

    s = it->second.stream;
    sz = it->second.bytes;

    alloc_map_.erase(it);

    auto sit = stream_map_.find(s);
    if (sit != stream_map_.end()) {
      sit->second.erase(p);
      if (sit->second.empty()) {
        stream_map_.erase(sit);
      }
    }

    in_use_bytes_ = (sz <= in_use_bytes_) ? (in_use_bytes_ - sz) : 0;
  }

  // Ordered free on the stream that allocated p
  CUDA_CALL_THROW(cudaFreeAsync(p, s));
}

void CudaMempoolArena::ReleaseStreamBuffers(Stream* stream) {
  const cudaStream_t s = ResolveCudaStream(stream);

  // Gather pointers to free while holding the lock for map updates.
  InlinedVector<void*> to_free;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto sit = stream_map_.find(s);
    if (sit == stream_map_.end()) {
      return;
    }

    to_free.reserve(sit->second.size());
    for (void* p : sit->second) {
      // Remove from alloc_map_ and adjust stats.
      auto ait = alloc_map_.find(p);
      if (ait != alloc_map_.end()) {
        const size_t sz = ait->second.bytes;
        in_use_bytes_ = (sz <= in_use_bytes_) ? (in_use_bytes_ - sz) : 0;
        to_free.push_back(p);
        alloc_map_.erase(ait);
      }
    }
    stream_map_.erase(sit);
  }

  for (void* p : to_free) {
    CUDA_CALL_THROW(cudaFreeAsync(p, s));
  }
}

Status CudaMempoolArena::Shrink() {
  // Trim the pool; live allocations are not affected.
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemPoolTrimTo(pool_, bytes_to_keep_)));

  // Right-size maps under lock.
  std::lock_guard<std::mutex> lock(mutex_);
  MaybeRehashLocked();
  ++num_arena_shrinkages_;
  return Status::OK();
}

void CudaMempoolArena::GetStats(AllocatorStats* stats) {
  if (!stats) return;
  std::lock_guard<std::mutex> lock(mutex_);
  stats->num_allocs = num_allocs_;
  stats->total_allocated_bytes = total_allocated_;
  stats->bytes_in_use = in_use_bytes_;
  stats->max_bytes_in_use = max_bytes_in_use_;
  stats->num_arena_shrinkages = num_arena_shrinkages_;
}

cudaStream_t CudaMempoolArena::ResolveCudaStream(Stream* stream) noexcept {
  if (!stream) return static_cast<cudaStream_t>(0);
  return static_cast<cudaStream_t>(stream->GetHandle());
}

void CudaMempoolArena::MaybeRehashLocked() {
  const size_t alloc_sz = alloc_map_.size();
  const size_t stream_sz = stream_map_.size();
  if (alloc_sz > 0) alloc_map_.reserve(alloc_sz);
  if (stream_sz > 0) stream_map_.reserve(stream_sz);
}

void CudaMempoolArena::SyncAllKnownStreams_NoThrow() {
  for (const auto& kv : stream_map_) {
    const cudaStream_t s = kv.first;
    (void)cudaStreamSynchronize(s);  // ignore errors; device-wide sync follows
  }
}

}  // namespace cuda
}  // namespace onnxruntime

#endif  // ORT_CUDA_HAS_MEMPOOL_API
