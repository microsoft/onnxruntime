/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation
// Adapted from onnxruntime/test/autoep/library/example_plugin_ep/ep_arena.h
// for the CUDA plugin EP arena allocator.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda_allocator_plugin.h"

#if defined(PLATFORM_WINDOWS) || defined(_WIN32)
#include <intrin.h>
#endif

namespace onnxruntime {
namespace cuda_plugin {

// Type-erasing unique_ptr for raw OrtAllocator ownership.
// The factory creates the raw allocator with a deleter that knows the concrete type.
using AllocatorUniquePtr = std::unique_ptr<OrtAllocator, std::function<void(OrtAllocator*)>>;

enum ArenaExtendStrategy {
  kDefault = -1,
  kNextPowerOfTwo = 0,
  kSameAsRequested = 1,
};

// Copied from onnxruntime::OrtArenaCfg so the values and config key names match.
struct ArenaConfig {
  static const ArenaExtendStrategy DEFAULT_ARENA_EXTEND_STRATEGY = ArenaExtendStrategy::kNextPowerOfTwo;
  static const int DEFAULT_INITIAL_CHUNK_SIZE_BYTES = 1 * 1024 * 1024;
  static const int DEFAULT_MAX_DEAD_BYTES_PER_CHUNK = 128 * 1024 * 1024;
  static const int DEFAULT_INITIAL_GROWTH_CHUNK_SIZE_BYTES = 2 * 1024 * 1024;
  static const int64_t DEFAULT_MAX_POWER_OF_TWO_EXTEND_BYTES = 1024 * 1024 * 1024;  // 1GB
  static const size_t DEFAULT_MAX_MEM = std::numeric_limits<size_t>::max();

  ArenaConfig(size_t max_mem = std::numeric_limits<size_t>::max(),
              ArenaExtendStrategy arena_extend_strategy = DEFAULT_ARENA_EXTEND_STRATEGY,
              int initial_chunk_size_bytes = DEFAULT_INITIAL_CHUNK_SIZE_BYTES,
              int max_dead_bytes_per_chunk = DEFAULT_MAX_DEAD_BYTES_PER_CHUNK,
              int initial_growth_chunk_size_bytes = DEFAULT_INITIAL_GROWTH_CHUNK_SIZE_BYTES,
              int64_t max_power_of_two_extend_bytes = DEFAULT_MAX_POWER_OF_TWO_EXTEND_BYTES)
      : max_mem(max_mem),
        arena_extend_strategy(arena_extend_strategy),
        initial_chunk_size_bytes(initial_chunk_size_bytes),
        max_dead_bytes_per_chunk(max_dead_bytes_per_chunk),
        initial_growth_chunk_size_bytes(initial_growth_chunk_size_bytes),
        max_power_of_two_extend_bytes(max_power_of_two_extend_bytes) {
    if (arena_extend_strategy == ArenaExtendStrategy::kDefault) {
      arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo;
    }
  }

  size_t max_mem;
  ArenaExtendStrategy arena_extend_strategy;
  int initial_chunk_size_bytes;
  int max_dead_bytes_per_chunk;
  int initial_growth_chunk_size_bytes;
  int64_t max_power_of_two_extend_bytes;

  bool IsValid() const {
    return initial_chunk_size_bytes > 0 &&
           max_dead_bytes_per_chunk > 0 &&
           initial_growth_chunk_size_bytes > 0 &&
           max_power_of_two_extend_bytes > 0;
  }

  struct ConfigKeyNames {
    static constexpr const char* ArenaExtendStrategy = "arena.extend_strategy";
    static constexpr const char* InitialChunkSizeBytes = "arena.initial_chunk_size_bytes";
    static constexpr const char* MaxDeadBytesPerChunk = "arena.max_dead_bytes_per_chunk";
    static constexpr const char* InitialGrowthChunkSizeBytes = "arena.initial_growth_chunk_size_bytes";
    static constexpr const char* MaxPowerOfTwoExtendBytes = "arena.max_power_of_two_extend_bytes";
    static constexpr const char* MaxMem = "arena.max_mem";
  };

  static ArenaConfig FromKeyValuePairs(const OrtApi& api, const OrtKeyValuePairs& kvps) {
    ArenaConfig config{};
    const char* value = nullptr;

    if (value = api.GetKeyValue(&kvps, ConfigKeyNames::ArenaExtendStrategy); value) {
      config.arena_extend_strategy = std::string(value) == "1" ? kSameAsRequested : kNextPowerOfTwo;
    }

    if (value = api.GetKeyValue(&kvps, ConfigKeyNames::InitialChunkSizeBytes); value) {
      config.initial_chunk_size_bytes = std::stoi(std::string(value));
    }

    if (value = api.GetKeyValue(&kvps, ConfigKeyNames::MaxDeadBytesPerChunk); value) {
      config.max_dead_bytes_per_chunk = std::stoi(std::string(value));
    }

    if (value = api.GetKeyValue(&kvps, ConfigKeyNames::InitialGrowthChunkSizeBytes); value) {
      config.initial_growth_chunk_size_bytes = std::stoi(std::string(value));
    }

    if (value = api.GetKeyValue(&kvps, ConfigKeyNames::MaxPowerOfTwoExtendBytes); value) {
      config.max_power_of_two_extend_bytes = std::stoll(value);
    }

    if (value = api.GetKeyValue(&kvps, ConfigKeyNames::MaxMem); value) {
      config.max_mem = static_cast<size_t>(std::stoull(std::string(value)));
    }

    return config;
  }
};

// Macros used by ArenaImpl (adapted from plugin_ep_utils.h for CUDA plugin namespace).

#define CUDA_ARENA_ENFORCE(condition, ...)                \
  do {                                                    \
    if (!(condition)) {                                   \
      std::ostringstream oss;                             \
      oss << "CUDA_ARENA_ENFORCE failed: " << #condition; \
      oss << " " << __VA_ARGS__;                          \
      throw std::runtime_error(oss.str());                \
    }                                                     \
  } while (false)

#define CUDA_ARENA_LOG(level, ...)                                                                         \
  do {                                                                                                     \
    std::ostringstream ss;                                                                                 \
    ss << __VA_ARGS__;                                                                                     \
    OrtStatus* _log_status = api_.Logger_LogMessage(&logger_, ORT_LOGGING_LEVEL_##level, ss.str().c_str(), \
                                                    ORT_FILE, __LINE__, __FUNCTION__);                     \
    if (_log_status) api_.ReleaseStatus(_log_status);                                                      \
  } while (false)

#define CUDA_ARENA_RETURN_ERROR(code, ...)            \
  do {                                                \
    std::ostringstream ss;                            \
    ss << __VA_ARGS__;                                \
    return api_.CreateStatus(code, ss.str().c_str()); \
  } while (false)

// A memory allocator that implements a 'best-fit with coalescing' algorithm.
// This is essentially a very simple version of Doug Lea's malloc (dlmalloc).
//
// Adapted from the example plugin EP arena (ep_arena.h/cc).
class ArenaImpl {
 public:
  static const ArenaExtendStrategy DEFAULT_ARENA_EXTEND_STRATEGY = ArenaExtendStrategy::kNextPowerOfTwo;
  static const int DEFAULT_INITIAL_CHUNK_SIZE_BYTES = 1 * 1024 * 1024;
  static const int DEFAULT_MAX_DEAD_BYTES_PER_CHUNK = 128 * 1024 * 1024;
  static const int DEFAULT_INITIAL_GROWTH_CHUNK_SIZE_BYTES = 2 * 1024 * 1024;
  static const int64_t DEFAULT_MAX_POWER_OF_TWO_EXTEND_BYTES = 1024 * 1024 * 1024;  // 1GB
  static const size_t DEFAULT_MAX_MEM = std::numeric_limits<size_t>::max();

  ArenaImpl(AllocatorUniquePtr allocator, const ArenaConfig& config, const OrtApi& api,
            const OrtLogger& logger);

  ~ArenaImpl();

  void* Alloc(size_t size);
  void* AllocOnStream(size_t size, OrtSyncStream* stream);
  void Free(void* p);

  // Allocate memory directly. Used for initializers so they don't affect arena growth patterns.
  void* Reserve(size_t size);

  OrtStatus* GetStats(OrtKeyValuePairs** stats);

  size_t RequestedSize(const void* ptr);
  size_t AllocatedSize(const void* ptr);

  // Un-assign chunks that are currently assigned to the stream.
  // Called from OrtSyncStreamImpl::OnSessionRunEnd.
  OrtStatus* ResetChunksUsingStream(const OrtSyncStreamImpl* stream_impl);

 private:
  void* AllocateRawInternal(size_t num_bytes, OrtSyncStream* stream, bool dump_log_on_failure);
  void DeallocateRawInternal(void* ptr);

  using ChunkHandle = size_t;
  static const size_t kInvalidChunkHandle = static_cast<size_t>(-1);

  using BinNum = int;
  static const int kInvalidBinNum = -1;
  static const int kNumBins = 21;

  struct Chunk {
    size_t size = 0;
    size_t requested_size = 0;
    int64_t allocation_id = -1;
    void* ptr = nullptr;
    ChunkHandle prev = kInvalidChunkHandle;
    ChunkHandle next = kInvalidChunkHandle;
    BinNum bin_num = kInvalidBinNum;
    OrtSyncStream* stream = nullptr;
    uint64_t stream_sync_id = 0;

    bool in_use() const { return allocation_id != -1; }

    std::string DebugString(ArenaImpl* a, bool recurse) {
      std::ostringstream ss;
      ss << "  Size: " << size << " | Requested Size: " << requested_size << " | in_use: " << in_use();
      if (recurse && prev != ArenaImpl::kInvalidChunkHandle) {
        Chunk* p = a->ChunkFromHandle(prev);
        ss << ", prev: " << p->DebugString(a, false);
      }
      if (recurse && next != ArenaImpl::kInvalidChunkHandle) {
        Chunk* n = a->ChunkFromHandle(next);
        ss << ", next: " << n->DebugString(a, false);
      }
      return ss.str();
    }
  };

  struct Bin {
    size_t bin_size = 0;

    struct ChunkComparator {
      explicit ChunkComparator(ArenaImpl* allocator)
          : allocator_(allocator) {}

      bool operator()(const ChunkHandle ha, const ChunkHandle hb) const {
        const Chunk* a = allocator_->ChunkFromHandle(ha);
        const Chunk* b = allocator_->ChunkFromHandle(hb);
        if (a->size != b->size) {
          return a->size < b->size;
        }
        return a->ptr < b->ptr;
      }

     private:
      ArenaImpl* allocator_;
    };

    typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
    FreeChunkSet free_chunks;
    Bin(ArenaImpl* allocator, size_t bs)
        : bin_size(bs), free_chunks(ChunkComparator(allocator)) {}
  };

  static const size_t kMinAllocationBits = 8;
  static const size_t kMinAllocationSize = 1 << kMinAllocationBits;

  class AllocationRegion {
   public:
    AllocationRegion(void* ptr, size_t memory_size, int64_t id)
        : ptr_(ptr),
          memory_size_(memory_size),
          end_ptr_(static_cast<void*>(static_cast<char*>(ptr_) + memory_size_)),
          id_(id) {
      CUDA_ARENA_ENFORCE(0 == memory_size % kMinAllocationSize, __FUNCTION__);
      const size_t n_handles = (memory_size + kMinAllocationSize - 1) / kMinAllocationSize;
      handles_ = std::make_unique<ChunkHandle[]>(n_handles);
      for (size_t i = 0; i < n_handles; i++) {
        handles_[i] = kInvalidChunkHandle;
      }
    }

    AllocationRegion(AllocationRegion&& other) noexcept { Swap(other); }
    AllocationRegion() = default;
    ~AllocationRegion() = default;

    AllocationRegion& operator=(AllocationRegion&& other) noexcept {
      Swap(other);
      return *this;
    }

    void* ptr() const { return ptr_; }
    void* end_ptr() const { return end_ptr_; }
    size_t memory_size() const { return memory_size_; }
    int64_t id() const { return id_; }

    ChunkHandle get_handle(const void* p) const {
      return handles_[IndexFor(p)];
    }

    void set_handle(const void* p, ChunkHandle h) {
      handles_[IndexFor(p)] = h;
    }

    void erase(const void* p) {
      set_handle(p, kInvalidChunkHandle);
    }

   private:
    void Swap(AllocationRegion& other) {
      std::swap(ptr_, other.ptr_);
      std::swap(memory_size_, other.memory_size_);
      std::swap(end_ptr_, other.end_ptr_);
      std::swap(id_, other.id_);
      std::swap(handles_, other.handles_);
    }

    int IndexFor(const void* p) const {
      std::uintptr_t p_int = reinterpret_cast<std::uintptr_t>(p);
      std::uintptr_t base_int = reinterpret_cast<std::uintptr_t>(ptr_);
      CUDA_ARENA_ENFORCE(p_int >= base_int, "AllocationRegion::IndexFor");
      CUDA_ARENA_ENFORCE(p_int < base_int + memory_size_, "AllocationRegion::IndexFor");
      return static_cast<int>(((p_int - base_int) >> kMinAllocationBits));
    }

    void* ptr_ = nullptr;
    size_t memory_size_ = 0;
    void* end_ptr_ = nullptr;
    int64_t id_ = -1;
    std::unique_ptr<ChunkHandle[]> handles_;

    AllocationRegion& operator=(const AllocationRegion&) = delete;
  };

  class RegionManager {
   public:
    RegionManager() = default;
    ~RegionManager() = default;

    void AddAllocationRegion(void* ptr, size_t memory_size, int64_t id) {
      auto entry = std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      regions_.insert(entry, AllocationRegion(ptr, memory_size, id));
    }

    void RemoveAllocationRegion(void* ptr) {
      auto entry = std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      CUDA_ARENA_ENFORCE(entry != regions_.end(),
                         "RegionManager::RemoveAllocationRegion Could not find Region for: " << ptr);
      regions_.erase(entry);
    }

    ChunkHandle get_handle(const void* p) const {
      return RegionFor(p)->get_handle(p);
    }

    void set_handle(const void* p, ChunkHandle h) {
      return MutableRegionFor(p)->set_handle(p, h);
    }

    void erase(const void* p) { return MutableRegionFor(p)->erase(p); }

    const std::vector<AllocationRegion>& regions() const { return regions_; }

   private:
    RegionManager(const RegionManager&) = delete;
    RegionManager& operator=(const RegionManager&) = delete;
    RegionManager(RegionManager&&) = delete;
    RegionManager& operator=(RegionManager&&) = delete;

    static bool Comparator(const void* ptr, const AllocationRegion& other) {
      return ptr < other.end_ptr();
    }

    AllocationRegion* MutableRegionFor(const void* p) {
      return const_cast<AllocationRegion*>(RegionFor(p));
    }

    const AllocationRegion* RegionFor(const void* p) const {
      auto entry = std::upper_bound(regions_.begin(), regions_.end(), p, &Comparator);

      if (entry != regions_.end()) {
        return &(*entry);
      }

      CUDA_ARENA_ENFORCE(entry != regions_.end(),
                         "RegionManager::RegionFor Could not find Region for: " << p);
      return nullptr;
    }

   private:
    std::vector<AllocationRegion> regions_;
  };

  size_t RoundedBytes(size_t bytes);
  OrtStatus* Extend(size_t rounded_bytes);
  Chunk* FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes, OrtSyncStream* stream);
  void SplitChunk(ChunkHandle h, size_t num_bytes);
  void Merge(ChunkHandle h, ChunkHandle h2);
  void FreeAndMaybeCoalesce(ChunkHandle h);
  ChunkHandle Coalesce(ChunkHandle h);
  void InsertFreeChunkIntoBin(ChunkHandle h);
  void RemoveFreeChunkIterFromBin(Bin::FreeChunkSet* free_chunks,
                                  const Bin::FreeChunkSet::iterator& c);
  void RemoveFreeChunkFromBin(ChunkHandle h);
  Chunk* SplitFreeChunkFromBin(Bin::FreeChunkSet* free_chunks,
                               const Bin::FreeChunkSet::iterator& citer,
                               size_t rounded_bytes,
                               size_t num_bytes);
  void DeleteChunk(ChunkHandle h);
  void DumpMemoryLog(size_t num_bytes);
  ChunkHandle AllocateChunk();
  void DeallocateChunk(ChunkHandle h);
  Chunk* ChunkFromHandle(ChunkHandle h);

  struct BinDebugInfo {
    size_t total_bytes_in_use = 0;
    size_t total_bytes_in_bin = 0;
    size_t total_requested_bytes_in_use = 0;
    size_t total_chunks_in_use = 0;
    size_t total_chunks_in_bin = 0;
  };

  std::array<BinDebugInfo, kNumBins> GetBinDebugInfo();

  int Log2FloorNonZeroSlow(uint64_t n) {
    int r = 0;
    while (n > 0) {
      r++;
      n >>= 1;
    }
    return r - 1;
  }

  int Log2FloorNonZero(uint64_t n) {
#if defined(__GNUC__)
    return 63 ^ __builtin_clzll(n);
#elif defined(PLATFORM_WINDOWS) || defined(_WIN32)
    unsigned long index;
#if defined(_WIN64)
    _BitScanReverse64(&index, n);
#else
    auto high = static_cast<unsigned long>(n >> 32);
    if (_BitScanReverse(&index, high) > 0) {
      index += 32;
    } else {
      auto low = static_cast<unsigned long>((n << 32) >> 32);
      _BitScanReverse(&index, low);
    }
#endif
    return index;
#else
    return Log2FloorNonZeroSlow(n);
#endif
  }

  Bin* BinFromIndex(BinNum index) {
    return reinterpret_cast<Bin*>(&(bins_space_[index * sizeof(Bin)]));
  }

  size_t BinNumToSize(BinNum index) {
    return static_cast<size_t>(256) << index;
  }

  BinNum BinNumForSize(size_t bytes) {
    uint64_t v = std::max<size_t>(bytes, 256) >> kMinAllocationBits;
    int b = std::min(kNumBins - 1, Log2FloorNonZero(v));
    return b;
  }

  Bin* BinForSize(size_t bytes) {
    return BinFromIndex(BinNumForSize(bytes));
  }

  alignas(Bin) char bins_space_[sizeof(Bin) * kNumBins];

  mutable std::mutex lock_;

  AllocatorUniquePtr device_allocator_;
  const std::string allocator_name_;
  const ArenaConfig config_;

  RegionManager region_manager_;
  size_t curr_region_allocation_bytes_;

  int64_t next_allocation_id_;

  std::vector<Chunk> chunks_;
  ChunkHandle free_chunks_list_;
  std::unordered_map<void*, size_t> reserved_chunks_;

  std::unordered_map<const OrtSyncStream*, std::set<ChunkHandle>> stream_to_chunks_;
  std::unordered_map<const OrtSyncStreamImpl*, const OrtSyncStream*> impl_to_stream_;

  AllocatorStats stats_{};

  const OrtApi& api_;
  const OrtEpApi& ep_api_;
  const OrtLogger& logger_;

  ArenaImpl(const ArenaImpl&) = delete;
  ArenaImpl& operator=(const ArenaImpl&) = delete;
  ArenaImpl(ArenaImpl&&) = delete;
  ArenaImpl& operator=(ArenaImpl&&) = delete;
};

// CudaArenaAllocator wraps ArenaImpl and presents an OrtAllocator interface.
// Inherits from CudaAllocatorBase for uniform allocator handling.
class CudaArenaAllocator final : public CudaAllocatorBase {
 public:
  static OrtStatus* Create(CudaAllocatorKind kind,
                           const OrtMemoryInfo* memory_info,
                           AllocatorUniquePtr raw_allocator,
                           const OrtKeyValuePairs* options,
                           const OrtApi& api,
                           const OrtLogger& logger,
                           std::unique_ptr<CudaArenaAllocator>& out);

  CudaArenaAllocator(CudaAllocatorKind kind, const OrtMemoryInfo* memory_info,
                     std::unique_ptr<ArenaImpl> impl)
      : CudaAllocatorBase(kind, memory_info), impl_(std::move(impl)) {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Reserve = ReserveImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    GetStats = GetStatsImpl;
    // Stream-aware only for device arena, not pinned
    AllocOnStream = (kind == CudaAllocatorKind::kDevice) ? AllocOnStreamImpl : nullptr;
  }

  OrtStatus* ResetChunksUsingStream(const OrtSyncStreamImpl* stream_impl) {
    return impl_->ResetChunksUsingStream(stream_impl);
  }

 private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_, size_t size) noexcept {
    try {
      auto& arena = *static_cast<CudaArenaAllocator*>(this_);
      return arena.impl_->Alloc(size);
    } catch (...) {
      return nullptr;
    }
  }

  static void* ORT_API_CALL AllocOnStreamImpl(OrtAllocator* this_, size_t size, OrtSyncStream* stream) noexcept {
    try {
      auto& arena = *static_cast<CudaArenaAllocator*>(this_);
      return arena.impl_->AllocOnStream(size, stream);
    } catch (...) {
      return nullptr;
    }
  }

  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_, size_t size) noexcept {
    try {
      auto& arena = *static_cast<CudaArenaAllocator*>(this_);
      return arena.impl_->Reserve(size);
    } catch (...) {
      return nullptr;
    }
  }

  static void ORT_API_CALL FreeImpl(OrtAllocator* this_, void* p) noexcept {
    try {
      auto& arena = *static_cast<CudaArenaAllocator*>(this_);
      arena.impl_->Free(p);
    } catch (...) {
      // Swallow: exceptions must not propagate across C ABI boundary.
    }
  }

  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_) noexcept {
    const auto& arena = *static_cast<const CudaArenaAllocator*>(this_);
    return arena.GetMemoryInfo();
  }

  static OrtStatus* ORT_API_CALL GetStatsImpl(const OrtAllocator* this_, OrtKeyValuePairs** out) noexcept {
    try {
      const auto& arena = *static_cast<const CudaArenaAllocator*>(this_);
      return arena.impl_->GetStats(out);
    } catch (const std::exception& ex) {
      return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
    } catch (...) {
      return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION,
                                        "CudaArenaAllocator::GetStats failed with an unknown exception.");
    }
  }

  std::unique_ptr<ArenaImpl> impl_;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
