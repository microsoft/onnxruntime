// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_mempool_allocator_plugin.h"

#include <algorithm>
#include <sstream>
#include <string>

#include "core/common/common.h"

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
    auto parse_uint64 = [&](const char* key, uint64_t& out_val) -> OrtStatus* {
      const char* v = api.GetKeyValue(options, key);
      if (!v) return nullptr;
      const std::string sval(v);
      // std::stoull silently wraps negative values via strtoull.
      // Reject leading '-' so e.g. "-1" doesn't become a huge value.
      if (!sval.empty() && sval[0] == '-') {
        return api.CreateStatus(
            ORT_INVALID_ARGUMENT,
            (std::string("Negative value for ") + key + ": '" + v + "'").c_str());
      }
      OrtStatus* parse_status = nullptr;
      ORT_TRY {
        out_val = std::stoull(sval);
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          parse_status = api.CreateStatus(
              ORT_INVALID_ARGUMENT,
              (std::string("Invalid value for ") + key + ": '" + v + "' — " + ex.what())
                  .c_str());
        });
      }
      return parse_status;
    };

    OrtStatus* st = parse_uint64(ConfigKeyNames::PoolReleaseThreshold, pool_release_threshold);
    if (st) return st;

    uint64_t keep_val = 0;
    st = parse_uint64(ConfigKeyNames::BytesToKeepOnShrink, keep_val);
    if (st) return st;
    bytes_to_keep_on_shrink = static_cast<size_t>(keep_val);
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
    if (cuda_err != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
    }
    return api.CreateStatus(
        ORT_NOT_IMPLEMENTED,
        "CUDA mempool requires CUDA runtime 11.2 or later.");
  }

  int cuda_driver_version = 0;
  cuda_err = cudaDriverGetVersion(&cuda_driver_version);
  if (cuda_err != cudaSuccess || cuda_driver_version < 11020) {
    if (cuda_err != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
    }
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
    static_cast<void>(cudaGetLastError());
    std::string msg = "cudaMemPoolCreate failed for device " + std::to_string(device_id) +
                      ": " + cudaGetErrorName(cuda_err) + ": " + cudaGetErrorString(cuda_err);
    return api.CreateStatus(ORT_EP_FAIL, msg.c_str());
  }

  if (pool_release_threshold != 0) {
    cuda_err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                       &pool_release_threshold);
    if (cuda_err != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      cudaMemPoolDestroy(pool);
      std::string msg = "cudaMemPoolSetAttribute(ReleaseThreshold) failed: " +
                        std::string(cudaGetErrorName(cuda_err));
      return api.CreateStatus(ORT_EP_FAIL, msg.c_str());
    }
  }

  out = std::unique_ptr<CudaMempoolOrtAllocator>(
      new CudaMempoolOrtAllocator(memory_info, api, logger, pool, device_id,
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
                                                 int device_id,
                                                 uint64_t pool_release_threshold,
                                                 size_t bytes_to_keep_on_shrink)
    : CudaAllocatorBase(CudaAllocatorKind::kDevice, memory_info),
      ort_api_(api),
      logger_(logger),
      device_id_(device_id),
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
  Shrink = ShrinkImpl;
}

CudaMempoolOrtAllocator::~CudaMempoolOrtAllocator() {
  // Ensure we target the correct GPU — cudaDeviceSynchronize() and the default
  // stream are per-current-device, not per-pool.
  int prev_device = -1;
  const bool restore = cudaGetDevice(&prev_device) == cudaSuccess;
  ORT_IGNORE_RETURN_VALUE(cudaSetDevice(device_id_));

  // Enqueue frees for any remaining allocations on their recorded streams.
  for (auto& [ptr, rec] : alloc_map_) {
    ORT_IGNORE_RETURN_VALUE(cudaFreeAsync(ptr, rec.stream));
  }

  SyncAllKnownStreams();
  alloc_map_.clear();
  stream_map_.clear();

  // Safety barrier: SyncAllKnownStreams() only synchronizes streams tracked in
  // stream_map_. If any allocation was made visible to a stream not tracked here
  // (e.g., via cudaMemPoolExportPointer or external code passing the pointer to
  // another stream), those operations would not be captured. cudaDeviceSynchronize()
  // ensures all such untracked work completes before we trim/destroy the pool.
  ORT_IGNORE_RETURN_VALUE(cudaDeviceSynchronize());

  if (pool_) {
    // Destructor always trims to 0 — the pool is about to be destroyed.
    // bytes_to_keep_on_shrink_ is for the explicit Shrink() path, not teardown.
    ORT_IGNORE_RETURN_VALUE(cudaMemPoolTrimTo(pool_, 0));
    ORT_IGNORE_RETURN_VALUE(cudaMemPoolDestroy(pool_));
    pool_ = nullptr;
  }

  if (restore) {
    ORT_IGNORE_RETURN_VALUE(cudaSetDevice(prev_device));
  }
}

void* CudaMempoolOrtAllocator::AllocInternal(size_t size, cudaStream_t stream) {
  void* p = nullptr;
  cudaError_t err = cudaMallocFromPoolAsync(&p, size, pool_, stream);
  if (err != cudaSuccess) {
    // Return nullptr for all CUDA errors — ORT_THROW would abort() under
    // ORT_NO_EXCEPTIONS, and exceptions must not propagate across the C ABI
    // boundary from the noexcept Alloc/AllocOnStream callbacks.
    std::string msg = std::string("CudaMempoolOrtAllocator: cudaMallocFromPoolAsync failed: ") +
                      cudaGetErrorName(err) + ": " + cudaGetErrorString(err) +
                      ", size=" + std::to_string(size);
    LogMessage(ort_api_, logger_, ORT_LOGGING_LEVEL_ERROR, msg.c_str());
    return nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    alloc_map_.emplace(p, AllocationRecord{size, stream});
    stream_map_[stream].insert(p);

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

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4702)  // unreachable code — required for ORT_NO_EXCEPTIONS builds
#endif

/*static*/
void* ORT_API_CALL CudaMempoolOrtAllocator::AllocImpl(OrtAllocator* this_, size_t size) noexcept {
  if (size == 0) return nullptr;
  ORT_TRY {
    auto& self = *static_cast<CudaMempoolOrtAllocator*>(this_);
    constexpr cudaStream_t kDefaultStream = static_cast<cudaStream_t>(0);
    // The legacy default stream (NULL / 0) is per-current-device. Ensure we
    // target the correct GPU so the allocation lands on the pool's device.
    int prev_device = -1;
    const bool restore = cudaGetDevice(&prev_device) == cudaSuccess;
    if (cudaSetDevice(self.device_id_) != cudaSuccess) {
      if (restore) cudaSetDevice(prev_device);
      return nullptr;
    }
    void* p = self.AllocInternal(size, kDefaultStream);
    if (restore) cudaSetDevice(prev_device);
    return p;
  }
  ORT_CATCH(...) {
    return nullptr;
  }
  return nullptr;
}

/*static*/
void* ORT_API_CALL CudaMempoolOrtAllocator::AllocOnStreamImpl(OrtAllocator* this_, size_t size,
                                                              OrtSyncStream* stream) noexcept {
  if (size == 0) return nullptr;
  ORT_TRY {
    auto& self = *static_cast<CudaMempoolOrtAllocator*>(this_);
    cudaStream_t s = self.ResolveCudaStream(stream);
    return self.AllocInternal(size, s);
  }
  ORT_CATCH(...) {
    return nullptr;
  }
  return nullptr;
}

/*static*/
void ORT_API_CALL CudaMempoolOrtAllocator::FreeImpl(OrtAllocator* this_, void* p) noexcept {
  if (!p) return;
  ORT_TRY {
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
  }
  ORT_CATCH(...) {
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
  ORT_TRY {
    const auto& self = *static_cast<const CudaMempoolOrtAllocator*>(this_);

    OrtKeyValuePairs* kvps = nullptr;
    self.ort_api_.CreateKeyValuePairs(&kvps);

    AllocatorStats stats{};
    {
      std::lock_guard<std::mutex> lock(self.mutex_);
      stats.num_allocs = static_cast<int64_t>(self.num_allocs_);
      stats.bytes_in_use = static_cast<int64_t>(self.in_use_bytes_);
      stats.max_bytes_in_use = static_cast<int64_t>(self.max_bytes_in_use_);
      stats.max_alloc_size = static_cast<int64_t>(self.max_alloc_size_);
      stats.num_arena_shrinkages = static_cast<int64_t>(self.num_arena_shrinkages_);
    }

    // TotalAllocated reflects memory currently reserved by the pool (held from the
    // driver), matching BFC arena semantics where it tracks region memory in use.
    size_t reserved = 0;
    if (cudaMemPoolGetAttribute(self.pool_, cudaMemPoolAttrReservedMemCurrent, &reserved) == cudaSuccess) {
      stats.total_allocated_bytes = static_cast<int64_t>(reserved);
    }

    stats.ToKeyValuePairs(self.ort_api_, kvps);
    *out = kvps;
    return nullptr;
  }
  ORT_CATCH(const std::exception& ex) {
    OrtStatus* err = nullptr;
    ORT_HANDLE_EXCEPTION([&]() {
      err = Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
    });
    return err;
  }
  ORT_CATCH(...) {
    return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION,
                                      "CudaMempoolOrtAllocator::GetStats failed.");
  }
  return nullptr;  // required for ORT_NO_EXCEPTIONS
}

/*static*/
OrtStatus* ORT_API_CALL CudaMempoolOrtAllocator::ShrinkImpl(OrtAllocator* this_) noexcept {
  ORT_TRY {
    auto& self = *static_cast<CudaMempoolOrtAllocator*>(this_);

    cudaError_t err = cudaMemPoolTrimTo(self.pool_, self.bytes_to_keep_on_shrink_);
    if (err != cudaSuccess) {
      std::string msg = std::string("cudaMemPoolTrimTo failed: ") +
                        cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, msg.c_str());
    }

    {
      std::ostringstream oss;

      size_t reserved_size = 0;
      if (cudaMemPoolGetAttribute(self.pool_, cudaMemPoolAttrReservedMemCurrent,
                                  &reserved_size) == cudaSuccess) {
        oss << "CudaMempoolOrtAllocator::Shrink: reserved size after trim: "
            << reserved_size << " bytes.";
      } else {
        oss << "CudaMempoolOrtAllocator::Shrink: pool trimmed; unable to query reserved size.";
      }
      LogMessage(self.ort_api_, self.logger_, ORT_LOGGING_LEVEL_INFO, oss.str().c_str());
    }

    {
      std::lock_guard<std::mutex> lock(self.mutex_);
      ++self.num_arena_shrinkages_;
    }

    return nullptr;
  }
  ORT_CATCH(const std::exception& ex) {
    OrtStatus* err = nullptr;
    ORT_HANDLE_EXCEPTION([&]() {
      err = Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
    });
    return err;
  }
  ORT_CATCH(...) {
    return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION,
                                      "CudaMempoolOrtAllocator::Shrink failed.");
  }
  return nullptr;  // required for ORT_NO_EXCEPTIONS
}

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

}  // namespace cuda_plugin
}  // namespace onnxruntime
