// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_mempool_allocator_plugin.h"

#include <algorithm>
#include <sstream>
#include <string>

namespace onnxruntime {
namespace cuda_plugin {

namespace {

void LogMessage(const OrtApi& api, const OrtLogger& logger,
                OrtLoggingLevel level, const char* msg) {
  OrtStatus* st = api.Logger_LogMessage(&logger, level, msg, ORT_FILE, __LINE__,
                                        "CudaMempoolOrtAllocator");
  if (st != nullptr) {
    api.ReleaseStatus(st);
  }
}

}  // namespace

// static
OrtStatus* CudaMempoolOrtAllocator::Create(const OrtMemoryInfo* memory_info,
                                           const OrtKeyValuePairs* options,
                                           const OrtApi& api,
                                           const OrtLogger& logger,
                                           std::unique_ptr<CudaMempoolOrtAllocator>& out) {
  // Parse config from options
  uint64_t pool_release_threshold = 0;
  size_t bytes_to_keep_on_shrink = 0;

  if (options) {
    const char* value = nullptr;

    if ((value = api.GetKeyValue(options, ConfigKeyNames::PoolReleaseThreshold)) != nullptr) {
      pool_release_threshold = std::stoull(std::string(value));
    }

    if ((value = api.GetKeyValue(options, ConfigKeyNames::BytesToKeepOnShrink)) != nullptr) {
      bytes_to_keep_on_shrink = static_cast<size_t>(std::stoull(std::string(value)));
    }
  }

  // Get device id from memory_info
  int device_id = 0;
  OrtStatus* status = api.MemoryInfoGetId(memory_info, &device_id);
  if (status != nullptr) {
    return status;
  }

  // Check CUDA version supports mempools (requires 11.2+)
  int cuda_rt_version = 0;
  cudaError_t cuda_err = cudaRuntimeGetVersion(&cuda_rt_version);
  if (cuda_err != cudaSuccess || cuda_rt_version < 11020) {
    return api.CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "CUDA mempool requires CUDA runtime 11.2 or later.");
  }

  int cuda_driver_version = 0;
  cuda_err = cudaDriverGetVersion(&cuda_driver_version);
  if (cuda_err != cudaSuccess || cuda_driver_version < 11020) {
    return api.CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "CUDA mempool requires CUDA driver 11.2 or later.");
  }

  // Create a process-local device memory pool
  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = device_id;

  cudaMemPool_t pool = nullptr;
  cuda_err = cudaMemPoolCreate(&pool, &props);
  if (cuda_err != cudaSuccess) {
    std::string msg = "cudaMemPoolCreate failed for device " + std::to_string(device_id) +
                      ": " + cudaGetErrorName(cuda_err) + ": " + cudaGetErrorString(cuda_err);
    return api.CreateStatus(ORT_EP_FAIL, msg.c_str());
  }

  if (pool_release_threshold != 0) {
    cuda_err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                       &pool_release_threshold);
    if (cuda_err != cudaSuccess) {
      cudaMemPoolDestroy(pool);
      std::string msg = "cudaMemPoolSetAttribute(ReleaseThreshold) failed: " +
                        std::string(cudaGetErrorName(cuda_err));
      return api.CreateStatus(ORT_EP_FAIL, msg.c_str());
    }
  }

  out = std::unique_ptr<CudaMempoolOrtAllocator>(
      new CudaMempoolOrtAllocator(memory_info, api, logger, pool,
                                  pool_release_threshold, bytes_to_keep_on_shrink));

  {
    std::ostringstream oss;
    oss << "CudaMempoolOrtAllocator created on device " << device_id
        << " with pool_release_threshold=" << pool_release_threshold
        << " bytes_to_keep_on_shrink=" << bytes_to_keep_on_shrink << ".";
    LogMessage(api, logger, ORT_LOGGING_LEVEL_INFO, oss.str().c_str());
  }

  return nullptr;
}

CudaMempoolOrtAllocator::CudaMempoolOrtAllocator(const OrtMemoryInfo* memory_info,
                                                 const OrtApi& api,
                                                 const OrtLogger& logger,
                                                 cudaMemPool_t pool,
                                                 uint64_t pool_release_threshold,
                                                 size_t bytes_to_keep_on_shrink)
    : CudaAllocatorBase(CudaAllocatorKind::kDevice, memory_info),
      ort_api_(api),
      logger_(logger),
      pool_(pool),
      pool_release_threshold_(pool_release_threshold),
      bytes_to_keep_on_shrink_(bytes_to_keep_on_shrink) {
  version = ORT_API_VERSION;
  Alloc = AllocImpl;
  AllocOnStream = AllocOnStreamImpl;
  Free = FreeImpl;
  Reserve = ReserveImpl;
  Info = InfoImpl;
  GetStats = GetStatsImpl;
}

CudaMempoolOrtAllocator::~CudaMempoolOrtAllocator() {
  // Enqueue frees for any remaining allocations on their recorded streams.
  for (auto& [ptr, rec] : alloc_map_) {
    ORT_IGNORE_RETURN_VALUE(cudaFreeAsync(ptr, rec.stream));
  }

  SyncAllKnownStreams();
  alloc_map_.clear();
  stream_map_.clear();

  // Safety barrier
  ORT_IGNORE_RETURN_VALUE(cudaDeviceSynchronize());

  if (pool_) {
    ORT_IGNORE_RETURN_VALUE(cudaMemPoolTrimTo(pool_, 0));
    ORT_IGNORE_RETURN_VALUE(cudaMemPoolDestroy(pool_));
    pool_ = nullptr;
  }
}

void* CudaMempoolOrtAllocator::AllocInternal(size_t size, cudaStream_t stream) {
  void* p = nullptr;
  cudaError_t err = cudaMallocFromPoolAsync(&p, size, pool_, stream);
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << "CudaMempoolOrtAllocator: cudaMallocFromPoolAsync failed: "
        << cudaGetErrorName(err) << ": " << cudaGetErrorString(err)
        << ", size=" << size;
    throw std::runtime_error(oss.str());
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    alloc_map_.emplace(p, AllocationRecord{size, stream});
    stream_map_[stream].insert(p);

    total_allocated_ += size;
    in_use_bytes_ += size;
    max_bytes_in_use_ = std::max(max_bytes_in_use_, in_use_bytes_);
    max_alloc_size_ = std::max(max_alloc_size_, size);
    ++num_allocs_;
  }

  return p;
}

cudaStream_t CudaMempoolOrtAllocator::ResolveCudaStream(OrtSyncStream* stream) const {
  if (!stream) return static_cast<cudaStream_t>(0);
  return static_cast<cudaStream_t>(ort_api_.SyncStream_GetHandle(stream));
}

void CudaMempoolOrtAllocator::SyncAllKnownStreams() noexcept {
  for (const auto& [stream, ptrs] : stream_map_) {
    ORT_IGNORE_RETURN_VALUE(cudaStreamSynchronize(stream));
  }
}

// --- OrtAllocator C callbacks ---

/*static*/
void* ORT_API_CALL CudaMempoolOrtAllocator::AllocImpl(OrtAllocator* this_, size_t size) noexcept {
  if (size == 0) return nullptr;
  try {
    auto& self = *static_cast<CudaMempoolOrtAllocator*>(this_);
    constexpr cudaStream_t kDefaultStream = static_cast<cudaStream_t>(0);
    void* p = self.AllocInternal(size, kDefaultStream);
    // Synchronize the default stream so the returned pointer is immediately usable.
    ORT_IGNORE_RETURN_VALUE(cudaStreamSynchronize(kDefaultStream));
    return p;
  } catch (...) {
    return nullptr;
  }
}

/*static*/
void* ORT_API_CALL CudaMempoolOrtAllocator::AllocOnStreamImpl(OrtAllocator* this_, size_t size,
                                                              OrtSyncStream* stream) noexcept {
  if (size == 0) return nullptr;
  try {
    auto& self = *static_cast<CudaMempoolOrtAllocator*>(this_);
    cudaStream_t s = self.ResolveCudaStream(stream);
    return self.AllocInternal(size, s);
  } catch (...) {
    return nullptr;
  }
}

/*static*/
void ORT_API_CALL CudaMempoolOrtAllocator::FreeImpl(OrtAllocator* this_, void* p) noexcept {
  if (!p) return;
  try {
    auto& self = *static_cast<CudaMempoolOrtAllocator*>(this_);

    cudaStream_t s = static_cast<cudaStream_t>(0);
    size_t sz = 0;

    {
      std::lock_guard<std::mutex> lock(self.mutex_);
      auto it = self.alloc_map_.find(p);
      if (it == self.alloc_map_.end()) {
        LogMessage(self.ort_api_, self.logger_, ORT_LOGGING_LEVEL_WARNING,
                   "CudaMempoolOrtAllocator::Free: pointer not found in allocation map; ignoring.");
        return;
      }

      s = it->second.stream;
      sz = it->second.bytes;
      self.alloc_map_.erase(it);

      auto sit = self.stream_map_.find(s);
      if (sit != self.stream_map_.end()) {
        sit->second.erase(p);
        if (sit->second.empty()) {
          self.stream_map_.erase(sit);
        }
      }

      self.in_use_bytes_ = (sz <= self.in_use_bytes_) ? (self.in_use_bytes_ - sz) : 0;
    }

    // Ordered free on the stream that allocated p
    cudaError_t err = cudaFreeAsync(p, s);
    if (err != cudaSuccess) {
      LogMessage(self.ort_api_, self.logger_, ORT_LOGGING_LEVEL_WARNING,
                 "CudaMempoolOrtAllocator::Free: cudaFreeAsync failed.");
    }
  } catch (...) {
    // Swallow: exceptions must not propagate across C ABI boundary.
  }
}

/*static*/
void* ORT_API_CALL CudaMempoolOrtAllocator::ReserveImpl(OrtAllocator* this_, size_t size) noexcept {
  // Reserve is implemented as Alloc — all memory is freed when the allocator is destroyed.
  return AllocImpl(this_, size);
}

/*static*/
const OrtMemoryInfo* ORT_API_CALL CudaMempoolOrtAllocator::InfoImpl(
    const OrtAllocator* this_) noexcept {
  const auto& self = *static_cast<const CudaMempoolOrtAllocator*>(this_);
  return self.GetMemoryInfo();
}

/*static*/
OrtStatus* ORT_API_CALL CudaMempoolOrtAllocator::GetStatsImpl(
    const OrtAllocator* this_, OrtKeyValuePairs** out) noexcept {
  try {
    const auto& self = *static_cast<const CudaMempoolOrtAllocator*>(this_);

    OrtKeyValuePairs* kvps = nullptr;
    self.ort_api_.CreateKeyValuePairs(&kvps);

    AllocatorStats stats{};
    {
      std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(self.mutex_));
      stats.num_allocs = static_cast<int64_t>(self.num_allocs_);
      stats.total_allocated_bytes = static_cast<int64_t>(self.total_allocated_);
      stats.bytes_in_use = static_cast<int64_t>(self.in_use_bytes_);
      stats.max_bytes_in_use = static_cast<int64_t>(self.max_bytes_in_use_);
      stats.max_alloc_size = static_cast<int64_t>(self.max_alloc_size_);
    }

    stats.ToKeyValuePairs(self.ort_api_, kvps);
    *out = kvps;
    return nullptr;
  } catch (const std::exception& ex) {
    return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
  } catch (...) {
    return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION,
                                      "CudaMempoolOrtAllocator::GetStats failed.");
  }
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
