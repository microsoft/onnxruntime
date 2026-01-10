// Copyright (c) Microsoft.
// Licensed under the MIT License.

#include "cuda_mempool_arena.h"

#include <algorithm>

#include "core/providers/cuda/shared_inc/cuda_call.h"  // ORT CudaCall helpers
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace cuda {

// ======== CudaMempoolArena ========

CudaMempoolArena::CudaMempoolArena(const OrtMemoryInfo& memory_info,
                                   uint64_t pool_release_threshold,
                                   size_t bytes_to_keep_on_shrink,
                                   const logging::Logger* logger)
    : IArena(memory_info),
      pool_release_threshold_(pool_release_threshold),
      bytes_to_keep_on_shrink_(bytes_to_keep_on_shrink),
      logger_(logger) {
  if (logger_ == nullptr) {
    logger_ = &::onnxruntime::logging::LoggingManager::DefaultLogger();
  }

  // Create a process-local device memory pool for device_id_.
  // 'cudaMemAllocationTypeDevice' (for cudaMemPoolProps.allocType) not clear when it is available

  cudaMemPoolProps props{};
  // Pinned is not the same as pinned allocator, cudaMemLocationTypeDevice actually does not exist
  // even though is present in some internet docs.
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;        // local to process
  props.location.type = cudaMemLocationTypeDevice;  // Device memory
  props.location.id = this->Info().device.Id();

  CUDA_CALL_THROW(cudaMemPoolCreate(&pool_, &props));

  if (pool_release_threshold_ != 0) {
    CUDA_CALL_THROW(cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold,
                                            &pool_release_threshold_));
  }

  LOGS(*logger_, INFO) << "CudaMempoolArena created on device " << this->Info().device.Id()
                       << " with pool_release_threshold=" << pool_release_threshold_
                       << " bytes_to_keep_on_shrink=" << bytes_to_keep_on_shrink_ << ".";

  // Intentionally DO NOT call cudaDeviceSetMemPool(device_id_, pool_);
  // All allocations explicitly target this pool via cudaMallocFromPoolAsync.
}

CudaMempoolArena::~CudaMempoolArena() {
  // 1) Best-effort: enqueue frees for any remaining allocations on their recorded streams.
  //    No locking by design: destruction implies no concurrent access.
  for (auto& kv : alloc_map_) {
    void* p = kv.first;
    const cudaStream_t s = kv.second.stream;
    ORT_IGNORE_RETURN_VALUE(cudaFreeAsync(p, s));  // ignore errors in destructor
  }

  // 2) Synchronize all streams we know about (those that ever held allocations).
  SyncAllKnownStreams_NoThrow();

  // Now it is safe to drop our bookkeeping.
  alloc_map_.clear();
  stream_map_.clear();

  // 3) Safety barrier: ensure any frees enqueued on destroyed/unknown streams are completed.
  ORT_IGNORE_RETURN_VALUE(cudaDeviceSynchronize());  // ignore errors in destructor

  // 4) Trim to zero and destroy the pool.
  if (pool_) {
    ORT_IGNORE_RETURN_VALUE(cudaMemPoolTrimTo(pool_, 0));  // best-effort
    ORT_IGNORE_RETURN_VALUE(cudaMemPoolDestroy(pool_));
    pool_ = nullptr;
  }
}

void* CudaMempoolArena::Alloc(size_t size) {
  if (size == 0) return nullptr;

  void* p = nullptr;
  constexpr const cudaStream_t kDefaultStream = static_cast<cudaStream_t>(0);
  cudaError_t err = cudaMallocFromPoolAsync(&p, size, pool_, kDefaultStream);
  if (err != cudaSuccess) {
    ORT_THROW("CudaMempoolArena::Alloc: cudaMallocFromPoolAsync failed: ",
              cudaGetErrorString(err), " (", static_cast<int>(err), "), size=", size);
  }

  LOGS(*logger_, VERBOSE) << "CudaMempoolArena::Alloc: allocated "
                          << size << " bytes at " << p << " on default stream.";

  // In case the default stream is busy.
  ORT_IGNORE_RETURN_VALUE(cudaStreamSynchronize(kDefaultStream));

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

  LOGS(*logger_, VERBOSE) << "CudaMempoolArena::AllocOnStream: allocated "
                          << size << " bytes at " << p << " on stream "
                          << reinterpret_cast<uintptr_t>(s) << ".";

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
      LOGS(*logger_, WARNING) << "CudaMempoolArena::Free: pointer "
                              << p << " not found in allocation map; ignoring.";
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

Status CudaMempoolArena::Shrink() {
  // Trim the pool; live allocations are not affected.
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemPoolTrimTo(pool_, bytes_to_keep_on_shrink_)));

  size_t current_in_use = 0;
  ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrUsedMemCurrent,
                                                            &current_in_use)));

  // Query current reserved size. cudaMemPoolAttrReservedMemCurrent
  size_t reserved_size = 0;
  if (CUDA_CALL(cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReservedMemCurrent,
                                        &reserved_size))
          .IsOK()) {
    LOGS(*logger_, INFO) << "CudaMempoolArena::Shrink: pool current_in_use: " << current_in_use
                         << " reserved size after trim : " << reserved_size << " bytes.";
  } else {
    LOGS(*logger_, INFO) << "CudaMempoolArena pool has been shrunk; unable to query reserved size.";
  }

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
    ORT_IGNORE_RETURN_VALUE(cudaStreamSynchronize(s));  // ignore errors; device-wide sync follows
  }
}

bool CudaMempoolArena::IsCudaVersionSupported() noexcept {
  int ort_cuda_rt_version = 0;
  cudaError_t cuda_status = cudaRuntimeGetVersion(&ort_cuda_rt_version);
  if (cuda_status != cudaSuccess) {
    return false;
  }

  if (ort_cuda_rt_version < 11020) {
    return false;
  }

  int ort_cuda_driver_version = 0;
  cuda_status = cudaDriverGetVersion(&ort_cuda_driver_version);
  if (cuda_status != cudaSuccess) {
    return false;
  }

  if (ort_cuda_driver_version < 11020) {
    return false;
  }

  // Check if the driver version supports the runtime version
  if (ort_cuda_rt_version >= 12000 && ort_cuda_driver_version < 12000) {
    return false;
  }

  if (ort_cuda_rt_version >= 13000 && ort_cuda_driver_version < 13000) {
    return false;
  }

  return true;
}

}  // namespace cuda
}  // namespace onnxruntime
