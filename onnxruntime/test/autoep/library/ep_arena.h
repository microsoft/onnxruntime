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

#pragma once
#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <set>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "ep_allocator.h"
#include "example_plugin_ep_utils.h"

#if defined(PLATFORM_WINDOWS)
#include <intrin.h>
#endif

enum ArenaExtendStrategy {
  kDefault = -1,
  kNextPowerOfTwo = 0,
  kSameAsRequested = 1,
};

// copied from onnxruntime::OrtArenaCfg so the values and config key names match
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

  bool IsValid() {
    return initial_chunk_size_bytes > 0 &&
           max_dead_bytes_per_chunk > 0 &&
           initial_growth_chunk_size_bytes > 0 &&
           max_power_of_two_extend_bytes > 0;
  }

  // config key names that we parse in FromKeyValuePairs
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

// A memory allocator that implements a 'best-fit with coalescing' algorithm.
// This is essentially a very simple version of Doug Lea's malloc (dlmalloc).
//
// The goal of this allocator is to support defragmentation via coalescing.
// One assumption we make is that the process using this allocator owns pretty much all of the memory, and that nearly
// all requests to allocate memory go through this interface.
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

  // allocate memory directly. this is used for initializers so they don't affect the arena growth patterns
  void* Reserve(size_t size);

  OrtStatus* GetStats(OrtKeyValuePairs** stats);

  size_t RequestedSize(const void* ptr);
  size_t AllocatedSize(const void* ptr);

  // Un-assign chunks that are currently assigned to the stream.
  //
  // This should be called from OrtSyncStreamImpl::OnSessionRunEnd.
  // A stream is used in one session at a time. When called from OnSessionRunEnd we know that the stream is done and
  // will not be performing any more operations on the data.
  //
  // We don't have a better way to know when it's safe to re-use a chunk in another stream given the actual memory
  // usage is asynchronous on the GPU side, and the code assigning memory is running on CPU prior to that.
  OrtStatus* ResetChunksUsingStream(const OrtSyncStreamImpl* stream_impl);

 private:
  void* AllocateRawInternal(size_t num_bytes, OrtSyncStream* stream, bool dump_log_on_failure);
  void DeallocateRawInternal(void* ptr);

  // A ChunkHandle is an index into the chunks_ vector in BFCAllocator
  // kInvalidChunkHandle means an invalid chunk
  using ChunkHandle = size_t;
  static const size_t kInvalidChunkHandle = static_cast<size_t>(-1);

  using BinNum = int;
  static const int kInvalidBinNum = -1;
  static const int kNumBins = 21;

  // Chunks point to memory.  Their prev/next pointers form a
  // doubly-linked list of addresses sorted by base address that
  // must be contiguous.  Chunks contain information about whether
  // they are in use or whether they are free, and contain a pointer
  // to the bin they are in.
  struct Chunk {
    size_t size = 0;  // Full size of buffer.

    // We sometimes give chunks that are larger than needed to reduce
    // fragmentation.  requested_size keeps track of what the client
    // actually wanted so we can understand whether our splitting
    // strategy is efficient.
    size_t requested_size = 0;

    // allocation_id is set to -1 when the chunk is not in use. It is assigned a
    // value greater than zero before the chunk is returned from
    // AllocateRaw, and this value is unique among values assigned by
    // the parent allocator.
    int64_t allocation_id = -1;
    void* ptr = nullptr;  // pointer to granted subbuffer.

    // If not kInvalidChunkHandle, the memory referred to by 'prev' is directly
    // preceding the memory used by this chunk.  E.g., It should start
    // at 'ptr - prev->size'
    ChunkHandle prev = kInvalidChunkHandle;

    // If not kInvalidChunkHandle, the memory referred to by 'next' is directly
    // following the memory used by this chunk.  E.g., It should be at
    // 'ptr + next->size'
    ChunkHandle next = kInvalidChunkHandle;

    // What bin are we in?
    BinNum bin_num = kInvalidBinNum;

    OrtSyncStream* stream = nullptr;
    // Current sync id of `stream` when it was assigned the Chunk.
    // If the chunk is assigned to a stream and is free, and another Stream wants to use it, that Stream must have
    // synchronized with `stream` at a sync id > to stream_sync_id.
    // stream_sync_id is set when the chunk is first assigned to `stream`.
    // The sync id is incremented at the start of sync, so any chunk with a previous sync id is safe to re-assign.
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

  // A Bin is a collection of similar-sized free chunks.
  struct Bin {
    // All chunks in this bin have >= bin_size memory.
    size_t bin_size = 0;

    struct ChunkComparator {
      explicit ChunkComparator(ArenaImpl* allocator)
          : allocator_(allocator) {}

      // Sort first by size and then use pointer address as a tie breaker.
      bool operator()(const ChunkHandle ha,
                      const ChunkHandle hb) const {
        const Chunk* a = allocator_->ChunkFromHandle(ha);
        const Chunk* b = allocator_->ChunkFromHandle(hb);
        if (a->size != b->size) {
          return a->size < b->size;
        }
        return a->ptr < b->ptr;
      }

     private:
      ArenaImpl* allocator_;  // The parent allocator
    };

    typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
    // List of free chunks within the bin, sorted by chunk size.
    // Chunk * not owned.
    FreeChunkSet free_chunks;
    Bin(ArenaImpl* allocator, size_t bs)
        : bin_size(bs), free_chunks(ChunkComparator(allocator)) {}
  };

  static const size_t kMinAllocationBits = 8;
  static const size_t kMinAllocationSize = 1 << kMinAllocationBits;

  // AllocationRegion maps pointers to ChunkHandles for a single
  // contiguous memory region.
  //
  // This class is thread-compatible.
  class AllocationRegion {
   public:
    AllocationRegion(void* ptr, size_t memory_size, int64_t id)
        : ptr_(ptr),
          memory_size_(memory_size),
          end_ptr_(static_cast<void*>(static_cast<char*>(ptr_) + memory_size_)),
          id_(id) {
      EP_ENFORCE(0 == memory_size % kMinAllocationSize, __FUNCTION__);

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
      EP_ENFORCE(p_int >= base_int, "AllocationRegion::IndexFor");
      EP_ENFORCE(p_int < base_int + memory_size_, "AllocationRegion::IndexFor");
      return static_cast<int>(((p_int - base_int) >> kMinAllocationBits));
    }

    // metadata about the allocation region.
    void* ptr_ = nullptr;
    size_t memory_size_ = 0;
    void* end_ptr_ = nullptr;
    // A unique identifier for this allocation region
    // (May be used by the client to track which allocation region was allocated first, second, and so on)
    int64_t id_ = -1;

    // Array of size "memory_size / kMinAllocationSize".  It is
    // indexed by (p-base) / kMinAllocationSize, contains ChunkHandle
    // for the memory allocation represented by "p"
    std::unique_ptr<ChunkHandle[]> handles_;

    AllocationRegion& operator=(const AllocationRegion&) = delete;
  };

  // RegionManager aggregates one or more "AllocationRegions" and provides
  // a layer of indirection from pointers to the underlying ChunkHandle,
  // allowing allocation across multiple discontiguous memory regions.
  //
  // This class is thread-compatible.
  class RegionManager {
   public:
    RegionManager() = default;
    ~RegionManager() = default;

    void AddAllocationRegion(void* ptr, size_t memory_size, int64_t id) {
      // Insert sorted by end_ptr
      auto entry = std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      regions_.insert(entry, AllocationRegion(ptr, memory_size, id));
    }

    void RemoveAllocationRegion(void* ptr) {
      auto entry = std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      EP_ENFORCE(entry != regions_.end(), "RegionManager::RemoveAllocationRegion Could not find Region for: " << ptr);
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

      EP_ENFORCE(entry != regions_.end(), "RegionManager::RegionFor Could not find Region for: " << p);
      return nullptr;
    }

   private:
    std::vector<AllocationRegion> regions_;
  };

  // Returns 'bytes' rounded up to the next highest kMinAllocationSize.
  size_t RoundedBytes(size_t bytes);

  // Try to add a new memory region that can satisfy an allocation of
  // 'rounded_bytes' bytes.
  OrtStatus* Extend(size_t rounded_bytes);

  // Returns an underlying allocated chunk of size
  // 'rounded_bytes'.
  ArenaImpl::Chunk* FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes, OrtSyncStream* stream);

  // Splits the chunk specified by 'h' into two chunks, one at least
  // of size 'num_bytes'.
  void SplitChunk(ChunkHandle h, size_t num_bytes);

  // Merges the two chunk handles.  Requires that the chunks are
  // contiguous in their allocation.
  void Merge(ChunkHandle h, ChunkHandle h2);

  // Frees the memory represented by 'h', coalescing the chunk if
  // possible.
  void FreeAndMaybeCoalesce(ChunkHandle h);

  ArenaImpl::ChunkHandle Coalesce(ChunkHandle h);

  // Adds the chunk 'h' to the proper free bin.
  void InsertFreeChunkIntoBin(ChunkHandle h);

  // Removes the free chunk pointed to by 'c' from the set free_chunks.
  void RemoveFreeChunkIterFromBin(Bin::FreeChunkSet* free_chunks,
                                  const Bin::FreeChunkSet::iterator& c);

  // Removes a free chunk from the bin.
  void RemoveFreeChunkFromBin(ChunkHandle h);

  ArenaImpl::Chunk* SplitFreeChunkFromBin(ArenaImpl::Bin::FreeChunkSet* free_chunks,
                                          const ArenaImpl::Bin::FreeChunkSet::iterator& citer,
                                          size_t rounded_bytes,
                                          size_t num_bytes);

  // Removes the chunk metadata represented by 'h'.
  void DeleteChunk(ChunkHandle h);

  void DumpMemoryLog(size_t num_bytes);

  ChunkHandle AllocateChunk();
  void DeallocateChunk(ChunkHandle h);

  Chunk* ChunkFromHandle(ChunkHandle h);

  // Information about a Bin that is useful for debugging.
  struct BinDebugInfo {
    size_t total_bytes_in_use = 0;
    size_t total_bytes_in_bin = 0;
    size_t total_requested_bytes_in_use = 0;
    size_t total_chunks_in_use = 0;
    size_t total_chunks_in_bin = 0;
  };

  // Computes and returns a BinDebugInfo for each Bin.
  std::array<BinDebugInfo, kNumBins> GetBinDebugInfo();

  int Log2FloorNonZeroSlow(uint64_t n) {
    int r = 0;
    while (n > 0) {
      r++;
      n >>= 1;
    }
    return r - 1;
  }

  // Returns floor(log2(n)).
  int Log2FloorNonZero(uint64_t n) {
#if defined(__GNUC__)
    return 63 ^ __builtin_clzll(n);
#elif defined(PLATFORM_WINDOWS)
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

  // Map from bin size to Bin
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

  // Counter containing the next unique identifier to assign to a newly-created chunk.
  int64_t next_allocation_id_;

  std::vector<Chunk> chunks_;
  ChunkHandle free_chunks_list_;  // Pointer to head of linked list of free Chunks
  std::unordered_map<void*, size_t> reserved_chunks_;

  // chunks being used by a stream
  std::unordered_map<const OrtSyncStream*, std::set<ChunkHandle>> stream_to_chunks_;

  // map to connect the OrtSyncStreamImpl the EP library creates to the OrtSyncStream that ORT uses.
  // we don't know that it's safe to re-use a chunk until the stream is done with, which is via the call to
  // OrtSyncStreamImpl::OnSessionRunEnd. the allocations see OrtSyncStream, so we need to connect things up to
  // un-assign chunks when StreamImpl::OnSessionRunEnd is called.
  std::unordered_map<const OrtSyncStreamImpl*, const OrtSyncStream*> impl_to_stream_;

  AllocatorStats stats_;

  const OrtApi& api_;
  const OrtEpApi& ep_api_;
  const OrtLogger& logger_;

  ArenaImpl(const ArenaImpl&) = delete;
  ArenaImpl& operator=(const ArenaImpl&) = delete;
  ArenaImpl(ArenaImpl&&) = delete;
  ArenaImpl& operator=(ArenaImpl&&) = delete;
};

struct ArenaAllocator : OrtAllocator {
  static OrtStatus* CreateOrtArenaAllocator(AllocatorUniquePtr allocator,
                                            const OrtKeyValuePairs* options,
                                            const OrtApi& api,
                                            const OrtLogger& logger,
                                            std::unique_ptr<ArenaAllocator>& arena_allocator) {
    ArenaConfig config = options ? ArenaConfig::FromKeyValuePairs(api, *options) : ArenaConfig{};
    const OrtMemoryInfo* mem_info = allocator->Info(allocator.get());
    auto impl = std::make_unique<ArenaImpl>(std::move(allocator), config, api, logger);

    arena_allocator = std::make_unique<ArenaAllocator>(std::move(impl), *mem_info);

    return nullptr;
  }

  ArenaAllocator(std::unique_ptr<ArenaImpl> implementation, const OrtMemoryInfo& memory_info)
      : impl_{std::move(implementation)},
        memory_info_{memory_info} {
    version = ORT_API_VERSION;
    Alloc = AllocImpl;
    Reserve = ReserveImpl;
    Free = FreeImpl;
    Info = InfoImpl;
    GetStats = GetStatsImpl;
    AllocOnStream = AllocOnStreamImpl;
  }

  // remove the OrtSyncStream* from any chunks that were using the stream
  OrtStatus* ResetChunksUsingStream(const OrtSyncStreamImpl* stream_impl) {
    impl_->ResetChunksUsingStream(stream_impl);
    return nullptr;
  }

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size) {
    auto& arena = *static_cast<ArenaAllocator*>(this_);
    return arena.impl_->Alloc(size);
  }

  static void* ORT_API_CALL AllocOnStreamImpl(struct OrtAllocator* this_, size_t size, OrtSyncStream* stream) {
    auto& arena = *static_cast<ArenaAllocator*>(this_);
    return arena.impl_->AllocOnStream(size, stream);
  }

  static void* ORT_API_CALL ReserveImpl(struct OrtAllocator* this_, size_t size) {
    auto& arena = *static_cast<ArenaAllocator*>(this_);
    return arena.impl_->Reserve(size);
  }

  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p) {
    auto& arena = *static_cast<ArenaAllocator*>(this_);
    arena.impl_->Free(p);
  }

  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_) {
    const auto& arena = *static_cast<const ArenaAllocator*>(this_);
    return &arena.memory_info_;
  }

  static OrtStatus* ORT_API_CALL GetStatsImpl(const struct OrtAllocator* this_, OrtKeyValuePairs** out) noexcept {
    const auto& arena = *static_cast<const ArenaAllocator*>(this_);
    return arena.impl_->GetStats(out);
  };

 private:
  std::unique_ptr<ArenaImpl> impl_;
  const OrtMemoryInfo& memory_info_;
};
