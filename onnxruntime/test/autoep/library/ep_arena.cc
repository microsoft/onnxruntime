// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_arena.h"
#include "example_plugin_ep_utils.h"

#include <cassert>
#include <map>

namespace onnxruntime {
namespace ep_utils {

#ifdef _WIN32
#define EP_WSTR(x) L##x
#define EP_FILE_INTERNAL(x) EP_WSTR(x)
#define EP_FILE EP_FILE_INTERNAL(__FILE__)
#else
#define EP_FILE __FILE__
#endif

#define LOG(level, ...)                                                                                             \
  do {                                                                                                              \
    std::ostringstream ss;                                                                                          \
    ss << __VA_ARGS__;                                                                                              \
    api_.Logger_LogMessage(&logger_, ORT_LOGGING_LEVEL_##level, ss.str().c_str(), EP_FILE, __LINE__, __FUNCTION__); \
  } while (false)

#define RETURN_ERROR(code, ...)                       \
  do {                                                \
    std::ostringstream ss;                            \
    ss << __VA_ARGS__;                                \
    return api_.CreateStatus(code, ss.str().c_str()); \
  } while (false)

#define THROW(...)       \
  std::ostringstream ss; \
  ss << __VA_ARGS__;     \
  throw new std::runtime_error(ss.str().c_str())

namespace {
std::string GetAllocatorName(const OrtApi& api, OrtAllocator& allocator) {
  const OrtMemoryInfo* mem_info = allocator.Info(&allocator);
  const char* allocator_name;
  auto* status = api.MemoryInfoGetName(mem_info, &allocator_name);  // never fails
  static_cast<void>(status);
  return allocator_name;
}
}  // namespace

ArenaImpl::ArenaImpl(std::unique_ptr<OrtAllocator> allocator, const ArenaConfig& config, const OrtApi& api,
                     const OrtLogger& logger)
    : device_allocator_{std::move(allocator)},
      allocator_name_{GetAllocatorName(api, *device_allocator_)},
      config_{config},
      api_{api},
      ep_api_{*api_.GetEpApi()},
      logger_{logger} {
  LOG(INFO, "Creating ArenaImpl for "
                << allocator_name_
                << " with following configs: initial_chunk_size_bytes: " << config_.initial_chunk_size_bytes
                << " max_dead_bytes_per_chunk: " << config_.max_dead_bytes_per_chunk
                << " initial_growth_chunk_size_bytes: " << config_.initial_growth_chunk_size_bytes
                << " max_power_of_two_extend_bytes: " << config_.max_power_of_two_extend_bytes
                << " memory limit: " << config_.max_mem
                << " arena_extend_strategy: " << config_.arena_extend_strategy);

  curr_region_allocation_bytes_ = RoundedBytes(std::min(config_.max_mem,
                                                        static_cast<size_t>(config_.initial_chunk_size_bytes)));
  // Allocate the requested amount of memory.
  ;
  stats_.bytes_limit = static_cast<int64_t>(config.max_mem);

  // We never want to shrink the initial allocation if the arena extend strategy is kNextPowerOfTwo.
  // This could seem confusingly arbitrary but the rationale is as follows:
  // The user selected initial allocation chunk is only valid for the arena extend strategy kNextPowerOfTwo
  // and the user has likely chosen this initial value so that any ad-hoc arena extensions/shrinkages could potentially
  // be avoided. So we do not consider the initial allocation for shrinkage whatever its usage status.
  // On the other hand, if the arena extension strategy is kSameAsRequested, any initial chunk set by the user or otherwise,
  // is moot and the arena will only extend based on the request size. In these cases, we consider any allocation for shrinkage
  // if it is left unused (even if it is the first allocation).
  if (config_.arena_extend_strategy == ArenaExtendStrategy::kSameAsRequested) {
    // Consider all allocation regions (including first allocation region) for shrinkage
    consider_first_allocation_region_for_shrinkage_ = true;
  } else {  // config_.arena_extend_strategy == kNextPowerOfTwo
    // Do not consider the first allocation region for shrinkage
    consider_first_allocation_region_for_shrinkage_ = false;
  }
  // Create a bunch of bins of various good sizes.

  // We create bins to fit all possible ranges that cover the
  // config_.max_mem starting from allocations up to 256 bytes to
  // allocations up to (and including) the memory limit.
  LOG(VERBOSE, "Creating " << kNumBins << " bins of max chunk size "
                           << BinNumToSize(0) << " to " << BinNumToSize(kNumBins - 1));

  for (BinNum b = 0; b < kNumBins; b++) {
    size_t bin_size = BinNumToSize(b);
    new (BinFromIndex(b)) Bin(this, bin_size);
    EP_ENFORCE((BinForSize(bin_size) == BinFromIndex(b) &&
                BinForSize(bin_size + 255) == BinFromIndex(b) &&
                BinForSize(bin_size * 2 - 1) == BinFromIndex(b)),
               "Invalid bin size for bin " << b);

    if (b + 1 < kNumBins) {
      EP_ENFORCE(BinForSize(bin_size * 2) != BinFromIndex(b), "Invalid bin size for " << b);
    }
  }
}

ArenaImpl::~ArenaImpl() {
  for (const auto& region : region_manager_.regions()) {
    device_allocator_->Free(device_allocator_.get(), region.ptr());
  }

  for (const auto& reserve_chunk : reserved_chunks_) {
    device_allocator_->Free(device_allocator_.get(), reserve_chunk.first);
  }

  for (BinNum b = 0; b < kNumBins; b++) {
    BinFromIndex(b)->~Bin();
  }
}

ArenaImpl::Chunk* ArenaImpl::ChunkFromHandle(ChunkHandle h) {
  EP_ENFORCE(h < chunks_.size(), "ChunkFromHandle");
  return &(chunks_[h]);
}

OrtStatus* ArenaImpl::Extend(size_t rounded_bytes) {
  size_t available_bytes = config_.max_mem - static_cast<size_t>(stats_.total_allocated_bytes);
  // Rounds available_bytes down to the nearest multiple of kMinAllocationSize.
  available_bytes = (available_bytes / kMinAllocationSize) * kMinAllocationSize;

  // Do we have enough space to handle the client's request?
  // If not, fail immediately.
  if (rounded_bytes > available_bytes) {
    RETURN_ERROR(ORT_EP_FAIL, "Available memory of " << available_bytes << " is smaller than requested bytes of "
                                                     << rounded_bytes);
  }

  auto safe_alloc = [this](size_t alloc_bytes) {
    void* new_mem = nullptr;
    try {
      new_mem = device_allocator_->Alloc(device_allocator_.get(), alloc_bytes);
    } catch (const std::bad_alloc&) {
      // attempted allocation can throw std::bad_alloc. we want to treat this the same as if it returned nullptr
      // so swallow the exception
    }
    // catch (const MyException& exception) {
    //   if your implementation threw, consider swallowing the exception to enable attempting a smaller allocation
    //   if possible
    //}
    return new_mem;
  };

  auto get_extend_bytes = [this, available_bytes](const size_t bytes, size_t& extend_bytes) -> OrtStatus* {
    extend_bytes = 0;
    if (config_.arena_extend_strategy == ArenaExtendStrategy::kNextPowerOfTwo) {
      // If curr_region_allocation_bytes_ is not enough to satisfy the
      // allocation, keep multiplying by a power of two until that is
      // sufficient.
      bool increased_allocation = false;
      while (bytes > curr_region_allocation_bytes_) {
        curr_region_allocation_bytes_ *= 2;
        increased_allocation = true;
      }

      extend_bytes = std::min(static_cast<size_t>(curr_region_allocation_bytes_), available_bytes);

      // we allocated the same number of bytes as the current region
      // the 2x is to double the minimum size of the next amount we'll allocate
      if (!increased_allocation) {
        if (config_.arena_extend_strategy == ArenaExtendStrategy::kNextPowerOfTwo &&
            static_cast<int64_t>(curr_region_allocation_bytes_) * 2 < config_.max_power_of_two_extend_bytes) {
          curr_region_allocation_bytes_ *= 2;
        } else {
          curr_region_allocation_bytes_ = config_.max_power_of_two_extend_bytes;
        }
      }
    } else if (config_.arena_extend_strategy == ArenaExtendStrategy::kSameAsRequested) {
      // BFC Arena could cause internal and external fragmentation. But, running training with
      // big batch size will be very sensitive to fragmentation. So, to avoid fragmentation,
      // just extend arena with actual requested size.
      extend_bytes = bytes;
    } else {
      RETURN_ERROR(ORT_INVALID_ARGUMENT, "Invalid arena extend strategy." << config_.arena_extend_strategy);
    }

    return nullptr;
  };

  size_t bytes;
  RETURN_IF_ERROR(get_extend_bytes(rounded_bytes, bytes));

  // Try allocating.
  void* mem_addr = safe_alloc(bytes);

  static constexpr float kBackpedalFactor = 0.9f;
  // Try allocating less memory.
  while (mem_addr == nullptr) {
    // kBackpedalFactor is float, bytes is size_t. The result of bytes * kBackpedalFactor is float. When we cast it to
    // size_t, which is a smaller type, it could loss data. This is what C4244 complains. The "static_cast<size_t>" here
    // is to suppress the warning. C26451 suggest we may change kBackpedalFactor to double to get better accuary. But if
    // we do that, AMD GPU CI build pipeline will have an "out-of-memory" error. So I choose to keep this piece of code
    // untouched and disable the warning first.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26451)
#endif
    bytes = RoundedBytes(static_cast<size_t>(bytes * kBackpedalFactor));
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
    // give up if we can't satisfy the requested size, or we're attempting an allocation of less than 8K.
    //
    // the latter protects against an infinite loop that occurs when bytes is less than 2560. at that point the 10%
    // reduction to 2304 bytes is undone by rounding to a 256 boundary in RoundedBytes, leading to an infinite loop.
    // the 8K value is just to give up a little earlier vs. getting all the way down to 2560 bytes.
    // If we can't allocate 8K, we're pretty much dead.
    if (bytes < rounded_bytes || bytes < 8 * 1024)
      break;

    mem_addr = safe_alloc(bytes);
  }

  if (mem_addr == nullptr) {
    RETURN_ERROR(ORT_EP_FAIL, "Failed to allocate memory for requested buffer of size " << rounded_bytes);
  }

  LOG(INFO, "Extended allocation by " << bytes << " bytes.");

  stats_.total_allocated_bytes += bytes;
  LOG(INFO, "Total allocated bytes: " << stats_.total_allocated_bytes);

  LOG(INFO, "Allocated memory at " << mem_addr << " to " << static_cast<void*>(static_cast<char*>(mem_addr) + bytes));

  region_manager_.AddAllocationRegion(mem_addr, bytes, stats_.num_arena_extensions);
  stats_.num_arena_extensions += 1;

  // Create one large chunk for the whole memory space that will
  // be chunked later.
  ChunkHandle h = AllocateChunk();
  ArenaImpl::Chunk* c = ChunkFromHandle(h);
  c->ptr = mem_addr;
  c->size = bytes;
  c->allocation_id = -1;
  c->prev = kInvalidChunkHandle;
  c->next = kInvalidChunkHandle;
  // assign the new created chunk to default stream, so it can be pick up by any stream
  c->stream = nullptr;

  region_manager_.set_handle(c->ptr, h);

  // TODO(vrv): Try to merge this new region with an existing region,
  // if the address space is contiguous, to avoid fragmentation
  // across regions.

  // Insert the chunk into the right bin.
  InsertFreeChunkIntoBin(h);

  return nullptr;
}

ArenaImpl::ChunkHandle
ArenaImpl::AllocateChunk() {
  if (free_chunks_list_ != kInvalidChunkHandle) {
    ChunkHandle h = free_chunks_list_;
    Chunk* c = ChunkFromHandle(h);
    free_chunks_list_ = c->next;
    return h;
  }
  ChunkHandle h = chunks_.size();
  chunks_.resize(h + 1);
  return h;
}

void ArenaImpl::DeallocateChunk(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);

  if (c->stream) {
    if (auto it = stream_to_chunks_.find(c->stream); it != stream_to_chunks_.end()) {
      size_t result = it->second.erase(h);
      static_cast<void>(result);  // doesn't matter if it wasn't found
    }

    c->stream = nullptr;
    c->stream_sync_id = 0;
  }

  c->next = free_chunks_list_;
  free_chunks_list_ = h;
}

// static
size_t ArenaImpl::RoundedBytes(size_t bytes) {
  return (kMinAllocationSize * ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));
}

void* ArenaImpl::Alloc(size_t size) {
  return AllocateRawInternal(size, nullptr, false);
}

void* ArenaImpl::AllocOnStream(size_t size, OrtSyncStream* stream) {
  if (stream_to_chunks_.find(stream) == stream_to_chunks_.end()) {
    stream_to_chunks_.insert({stream, std::set<size_t>{}});
  }

  return AllocateRawInternal(size, stream, false);
}

void* ArenaImpl::Reserve(size_t size) {
  if (size == 0)
    return nullptr;

  std::lock_guard<std::mutex> lock(lock_);

  LOG(INFO, "Reserving memory in ArenaImpl for " << allocator_name_ << " size: " << size);

  void* ptr = device_allocator_->Alloc(device_allocator_.get(), size);
  EP_ENFORCE(reserved_chunks_.find(ptr) == reserved_chunks_.end(), __FUNCTION__);
  reserved_chunks_.insert(std::pair<void*, size_t>(ptr, size));
  stats_.bytes_in_use += size;
  stats_.num_reserves += 1;
  stats_.num_allocs += 1;
  stats_.max_alloc_size = std::max<size_t>(static_cast<size_t>(stats_.max_alloc_size), size);
  stats_.max_bytes_in_use = std::max<int64_t>(static_cast<int64_t>(stats_.max_bytes_in_use), stats_.bytes_in_use);
  stats_.total_allocated_bytes += size;
  return ptr;
}

size_t ArenaImpl::RequestedSize(const void* ptr) {
  std::lock_guard<std::mutex> lock(lock_);
  ArenaImpl::ChunkHandle h = region_manager_.get_handle(ptr);
  EP_ENFORCE(h != kInvalidChunkHandle, __FUNCTION__);
  ArenaImpl::Chunk* c = ChunkFromHandle(h);
  return c->requested_size;
}

size_t ArenaImpl::AllocatedSize(const void* ptr) {
  std::lock_guard<std::mutex> lock(lock_);
  ArenaImpl::ChunkHandle h = region_manager_.get_handle(ptr);
  EP_ENFORCE(h != kInvalidChunkHandle, __FUNCTION__);
  ArenaImpl::Chunk* c = ChunkFromHandle(h);
  return c->size;
}

void* ArenaImpl::AllocateRawInternal(size_t num_bytes, OrtSyncStream* stream, bool dump_log_on_failure) {
  if (num_bytes == 0) {
    return nullptr;
  }

  // Round to multiple of kMinAllocationSize
  size_t rounded_bytes = RoundedBytes(num_bytes);

  // The BFC allocator tries to find the best fit first.
  BinNum bin_num = BinNumForSize(rounded_bytes);

  std::lock_guard<std::mutex> lock(lock_);
  // search for a valid chunk
  auto* chunk = FindChunkPtr(bin_num, rounded_bytes, num_bytes, stream);

  if (chunk != nullptr) {
    return chunk->ptr;
  }

  LOG(INFO, "Extending arena for " << allocator_name_
                                   << ". bin_num:" << bin_num << " (requested) num_bytes: " << num_bytes
                                   << " (actual) rounded_bytes:" << rounded_bytes);

  // Try to extend
  auto status = Extend(rounded_bytes);
  if (status == nullptr) {
    chunk = FindChunkPtr(bin_num, rounded_bytes, num_bytes, stream);
    if (chunk != nullptr) {
      return chunk->ptr;
    } else {
      status = api_.CreateStatus(ORT_EP_FAIL,
                                 ("Failed to find a free memory block despite calling Extend. rounded_bytes=" +
                                  std::to_string(rounded_bytes))
                                     .c_str());
    }
  }

  // We searched all bins for an existing free chunk to use and couldn't find one. Dump the memory log for analysis.
  if (dump_log_on_failure) {
    LOG(ERROR, "BFC Arena ran out of memory trying to allocate " << num_bytes);
    DumpMemoryLog(rounded_bytes);
  }

  throw new std::runtime_error(api_.GetErrorMessage(status));
}

OrtStatus* ArenaImpl::GetStats(OrtKeyValuePairs** stats) {
  std::lock_guard<std::mutex> lock(lock_);

  api_.CreateKeyValuePairs(stats);
  stats_.ToKeyValuePairs(api_, *stats);

  return nullptr;
}

ArenaImpl::Chunk* ArenaImpl::SplitFreeChunkFromBin(ArenaImpl::Bin::FreeChunkSet* free_chunks,
                                                   const ArenaImpl::Bin::FreeChunkSet::iterator& citer,
                                                   size_t rounded_bytes,
                                                   size_t num_bytes) {
  const ArenaImpl::ChunkHandle h = (*citer);
  RemoveFreeChunkIterFromBin(free_chunks, citer);
  ArenaImpl::Chunk* chunk = ChunkFromHandle(h);

  // If we can break the size of the chunk into two reasonably large pieces, do so.
  // In any case don't waste more than max_dead_bytes_per_chunk bytes on padding this alloc.
  if (chunk->size >= rounded_bytes * 2 ||
      static_cast<int64_t>(chunk->size - rounded_bytes) >= config_.max_dead_bytes_per_chunk) {
    SplitChunk(h, rounded_bytes);
    chunk = ChunkFromHandle(h);  // Update chunk pointer in case it moved
  }

  // The requested size of the returned chunk is what the user has allocated.
  chunk->requested_size = num_bytes;
  // Assign a unique id and increment the id counter, marking the chunk as being in use.
  chunk->allocation_id = next_allocation_id_++;

  ++stats_.num_allocs;
  stats_.bytes_in_use += chunk->size;
  stats_.max_bytes_in_use = std::max(stats_.max_bytes_in_use, stats_.bytes_in_use);
  stats_.max_alloc_size = std::max<int64_t>(stats_.max_alloc_size, static_cast<int64_t>(chunk->size));

  return chunk;
}

ArenaImpl::Chunk* ArenaImpl::FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes,
                                          OrtSyncStream* stream) {
  // First identify the first bin that could satisfy rounded_bytes.
  for (; bin_num < kNumBins; bin_num++) {
    // Start searching from the first bin for the smallest chunk that fits rounded_bytes.
    Bin* b = BinFromIndex(bin_num);
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end(); ++citer) {
      const ArenaImpl::ChunkHandle h = (*citer);
      ArenaImpl::Chunk* chunk = ChunkFromHandle(h);
      EP_ENFORCE(!chunk->in_use(), __FUNCTION__);

      if (chunk->size >= rounded_bytes) {
        // We found an existing chunk that fits us that wasn't in use.
        // If it's assigned to another stream, and we have synchronized with that stream more recently than it
        // was assigned, we can take the chunk.
        bool safe_to_use = chunk->stream == stream ||
                           !chunk->stream ||
                           (stream && chunk->stream &&
                            chunk->stream_sync_id < ep_api_.GetSyncIdForLastWaitOnSyncStream(chunk->stream, stream));

        if (safe_to_use) {
          chunk = SplitFreeChunkFromBin(&b->free_chunks, citer, rounded_bytes, num_bytes);

          if (stream) {
            chunk->stream = stream;
            chunk->stream_sync_id = ep_api_.SyncStream_GetSyncId(stream);
            stream_to_chunks_[stream].insert(h);
          }

          return chunk;
        }
      }
    }
  }

  return nullptr;
}

void ArenaImpl::SplitChunk(ArenaImpl::ChunkHandle h, size_t num_bytes) {
  // Allocate the new chunk before we do any ChunkFromHandle
  ChunkHandle h_new_chunk = AllocateChunk();

  Chunk* c = ChunkFromHandle(h);
  EP_ENFORCE(!c->in_use() && (c->bin_num == kInvalidBinNum), __FUNCTION__);

  // Create a new chunk starting num_bytes after c
  ArenaImpl::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
  new_chunk->stream = c->stream;
  new_chunk->stream_sync_id = c->stream_sync_id;

  new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
  region_manager_.set_handle(new_chunk->ptr, h_new_chunk);

  // Set the new sizes of the chunks.
  new_chunk->size = c->size - num_bytes;
  c->size = num_bytes;

  // The new chunk is not in use.
  new_chunk->allocation_id = -1;

  // Maintain the pointers.
  // c <-> c_neighbor becomes
  // c <-> new_chunk <-> c_neighbor
  ArenaImpl::ChunkHandle h_neighbor = c->next;
  new_chunk->prev = h;
  new_chunk->next = h_neighbor;
  c->next = h_new_chunk;
  if (h_neighbor != kInvalidChunkHandle) {
    Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
    c_neighbor->prev = h_new_chunk;
  }

  // Add the newly free chunk to the free bin.
  InsertFreeChunkIntoBin(h_new_chunk);
}

void ArenaImpl::Free(void* p) {
  if (p == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(lock_);
  auto it = reserved_chunks_.find(p);
  if (it != reserved_chunks_.end()) {
    device_allocator_->Free(device_allocator_.get(), it->first);
    stats_.bytes_in_use -= it->second;
    stats_.total_allocated_bytes -= it->second;
    reserved_chunks_.erase(it);
  } else {
    DeallocateRawInternal(p);
  }
}

void ArenaImpl::DeallocateRawInternal(void* ptr) {
  // Find the chunk from the ptr.
  ArenaImpl::ChunkHandle h = region_manager_.get_handle(ptr);
  EP_ENFORCE(h != kInvalidChunkHandle, __FUNCTION__);

  // Consider coalescing it.
  FreeAndMaybeCoalesce(h);
}

// Merges Chunk(h2) into Chunk(h1) when Chunk(h1)->next is h2 and Chunk(h2)->prev is h1.
void ArenaImpl::Merge(ArenaImpl::ChunkHandle h1,
                      ArenaImpl::ChunkHandle h2) {
  Chunk* c1 = ChunkFromHandle(h1);
  Chunk* c2 = ChunkFromHandle(h2);
  // We can only merge chunks that are not in use.
  EP_ENFORCE(!c1->in_use() && !c2->in_use() && c1->stream == c2->stream, __FUNCTION__);

  // c1's prev doesn't change, still points to the same ptr, and is
  // still not in use.

  // Fix up neighbor pointers
  //
  // c1 <-> c2 <-> c3 should become
  // c1 <-> c3

  ArenaImpl::ChunkHandle h3 = c2->next;
  c1->next = h3;
  EP_ENFORCE(c2->prev == h1, __FUNCTION__);
  if (h3 != kInvalidChunkHandle) {
    ArenaImpl::Chunk* c3 = ChunkFromHandle(h3);
    c3->prev = h1;
  }

  // Set the new size
  c1->size += c2->size;

  // we only merge chunks that have the same stream
  assert(c1->stream == c2->stream);
  c1->stream_sync_id = std::max(c1->stream_sync_id, c2->stream_sync_id);

  DeleteChunk(h2);
}

void ArenaImpl::DeleteChunk(ChunkHandle h) {
  // Delete h and cleanup all state
  Chunk* c = ChunkFromHandle(h);
  region_manager_.erase(c->ptr);
  DeallocateChunk(h);
}

void ArenaImpl::InsertFreeChunkIntoBin(ArenaImpl::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  EP_ENFORCE(!c->in_use() && (c->bin_num == kInvalidBinNum), __FUNCTION__);
  BinNum bin_num = BinNumForSize(c->size);
  Bin* new_bin = BinFromIndex(bin_num);
  c->bin_num = bin_num;
  new_bin->free_chunks.insert(h);
}

void ArenaImpl::RemoveFreeChunkIterFromBin(ArenaImpl::Bin::FreeChunkSet* free_chunks,
                                           const ArenaImpl::Bin::FreeChunkSet::iterator& citer) {
  ChunkHandle h = *citer;
  Chunk* c = ChunkFromHandle(h);
  EP_ENFORCE(!c->in_use() && (c->bin_num != kInvalidBinNum), __FUNCTION__);
  free_chunks->erase(citer);
  c->bin_num = kInvalidBinNum;
}

void ArenaImpl::RemoveFreeChunkFromBin(ArenaImpl::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  EP_ENFORCE(!c->in_use() && (c->bin_num != kInvalidBinNum), __FUNCTION__);
  EP_ENFORCE(BinFromIndex(c->bin_num)->free_chunks.erase(h) > 0, "Could not find chunk in bin");
  c->bin_num = kInvalidBinNum;
}

void ArenaImpl::FreeAndMaybeCoalesce(ArenaImpl::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  EP_ENFORCE(c->in_use() && (c->bin_num == kInvalidBinNum), __FUNCTION__);

  // Mark the chunk as no longer in use
  c->allocation_id = -1;

  // Updates the stats.
  stats_.bytes_in_use -= c->size;

  // This chunk is no longer in-use, consider coalescing the chunk
  // with adjacent chunks.
  ChunkHandle chunk_to_reassign = Coalesce(h);
  InsertFreeChunkIntoBin(chunk_to_reassign);
}

ArenaImpl::ChunkHandle ArenaImpl::Coalesce(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  EP_ENFORCE(!c->in_use(), __FUNCTION__);

  // This chunk is no longer in-use, consider coalescing the chunk with adjacent chunks.
  ChunkHandle chunk_to_reassign = h;

  // If the next chunk is free, coalesce the two
  if (c->next != kInvalidChunkHandle) {
    Chunk* cnext = ChunkFromHandle(c->next);
    // only merge the chunks belong to the same stream
    if (!cnext->in_use() && cnext->stream == c->stream) {
      chunk_to_reassign = h;

      // Deletes c->next
      RemoveFreeChunkFromBin(c->next);
      Merge(h, ChunkFromHandle(h)->next);
    }
  }

  // If the previous chunk is free, coalesce the two
  c = ChunkFromHandle(h);
  if (c->prev != kInvalidChunkHandle) {
    Chunk* cprev = ChunkFromHandle(c->prev);
    // only merge the chunks belong to the same stream
    if (!cprev->in_use() && cprev->stream == c->stream) {
      chunk_to_reassign = c->prev;

      RemoveFreeChunkFromBin(c->prev);  // this deletes c
      Merge(ChunkFromHandle(h)->prev, h);
    }
  }

  return chunk_to_reassign;
}

std::array<ArenaImpl::BinDebugInfo, ArenaImpl::kNumBins> ArenaImpl::GetBinDebugInfo() {
  std::array<BinDebugInfo, kNumBins> bin_infos;

  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      BinNum bin_num = BinNumForSize(c->size);
      BinDebugInfo& bin_info = bin_infos[bin_num];
      bin_info.total_bytes_in_bin += c->size;
      bin_info.total_chunks_in_bin++;

      if (c->in_use()) {
        bin_info.total_bytes_in_use += c->size;
        bin_info.total_requested_bytes_in_use += c->requested_size;
        bin_info.total_chunks_in_use++;
      } else {
        Bin* bin = BinFromIndex(bin_num);
        EP_ENFORCE(bin->free_chunks.count(h) == 1 && c->bin_num == bin_num, __FUNCTION__);
      }

      h = c->next;
    }
  }
  return bin_infos;
}

void ArenaImpl::DumpMemoryLog(size_t num_bytes) {
  const std::array<BinDebugInfo, kNumBins> bin_infos = GetBinDebugInfo();
  LOG(INFO, "Allocator:" << allocator_name_);
  LOG(INFO, "Bin size: Chunks in_use/total (if not zero). Allocated bytes in_use/total. Requested bytes.");

  size_t waste = 0;
  for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++) {
    Bin* b = BinFromIndex(bin_num);
    const BinDebugInfo& bin_info = bin_infos[bin_num];
    EP_ENFORCE(b->free_chunks.size() == bin_info.total_chunks_in_bin - bin_info.total_chunks_in_use, __FUNCTION__);

    if (bin_info.total_chunks_in_bin > 0) {
      LOG(INFO, b->bin_size
                    << ": Chunks " << bin_info.total_chunks_in_use << "/" << bin_info.total_chunks_in_bin
                    << ". Bytes "
                    << bin_info.total_bytes_in_use << "/" << bin_info.total_bytes_in_bin << ". "
                    << "Requested " << bin_info.total_requested_bytes_in_use << ".");

      waste += bin_info.total_bytes_in_use - bin_info.total_requested_bytes_in_use;
    }
  }

  if (waste > 0) {
    LOG(INFO, "Diff between in-use and requested bytes is " << waste);
  }

  // Find the bin that we would have liked to allocate in, so we can get some further analysis about fragmentation.
  Bin* b = BinForSize(num_bytes);

  LOG(INFO, "Bin for " << num_bytes
                       << " bytes has max bytes of " << b->bin_size
                       << ", Chunk State: ");

  for (ChunkHandle h : b->free_chunks) {
    Chunk* c = ChunkFromHandle(h);
    LOG(INFO, "  " << c->DebugString(this, true));
  }

  // Next show the chunks that are in use, and also summarize their number by size.
  LOG(INFO, "Overall chunks summary:");
  std::map<size_t, int> in_use_by_size;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        in_use_by_size[c->size]++;
      }
      LOG(INFO, (c->in_use() ? "  Chunk" : "  Free ") << " at " << c->ptr
                                                      << " of size " << c->size);
      h = c->next;
    }
  }

  LOG(INFO, "Summary of in-use chunks by size: ");
  size_t total_bytes = 0;
  for (auto& it : in_use_by_size) {
    LOG(INFO, "  " << it.second << " chunks of size " << it.first
                   << ". Total " << it.first * it.second);
    total_bytes += (it.first * it.second);
  }

  LOG(INFO, "Sum Total of in-use chunks: " << total_bytes);
  LOG(INFO, "Stats: \n"
                << stats_.DebugString());
}

void ArenaImpl::ResetChunksUsingStream(const OrtSyncStreamImpl* stream) {
  std::lock_guard<std::mutex> lock(lock_);

  auto it = impl_to_stream_.find(stream);
  if (it == impl_to_stream_.end()) {
      return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                "ResetChunksUsingStream called with unknown stream");
  }

  auto it = stream_to_chunks_.find(stream);
  if (it != stream_to_chunks_.end()) {
    const auto& chunk_handles = it->second;
    for (size_t handle : chunk_handles) {
      Chunk* c = ChunkFromHandle(handle);
      assert(c->stream == stream);  // something is out of sync if this is not the case
      // NOTE: The buffer may still be in-use, but the inference session that the stream was associated with is
      // done and will no longer be utilizing the buffer.
      // e.g. pre-allocated inference session output provided by the user was allocated using this arena.
      c->stream = nullptr;
    }

    stream_to_chunks_.erase(it);
  }

  // It's also possible to find the chunks this way, but that requires iterating every single in-use allocation.
  // We also repeat this for every single stream used in a session.
  // OTOH there's a cost to create/update keep streams_to_chunks_.
  // Using streams_to_chunks_ for now. It also simplifies debugging to have that info. If you're unsure about this
  // choice feel free to perf test the two approaches.
  //
  // for (const auto& region : region_manager_.regions()) {
  //   ChunkHandle region_begin_chunk = region_manager_.get_handle(region.ptr());
  //   ChunkHandle h = region_begin_chunk;
  //   while (h != kInvalidChunkHandle) {
  //     Chunk* c = ChunkFromHandle(h);
  //     if (c->stream == target_stream) {
  //       c->stream = nullptr;
  //       c->stream_sync_id = 0;
  //     }
  //     h = c->next;
  //   }
  // }

  // coalesce
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle region_begin_chunk = region_manager_.get_handle(region.ptr());
    ChunkHandle h = region_begin_chunk;
    while (h != kInvalidChunkHandle) {
      Chunk* c = ChunkFromHandle(h);
      if (!c->in_use()) {
        RemoveFreeChunkFromBin(h);
        ChunkHandle h_next = c->next;
        Chunk* c_next = h_next != kInvalidChunkHandle ? ChunkFromHandle(h_next) : nullptr;

        // merge until next chunk is different stream
        while (c_next && !c_next->in_use() && c_next->stream == c->stream) {
          Coalesce(h);
          h_next = c->next;
          c_next = h_next != kInvalidChunkHandle ? ChunkFromHandle(h_next) : nullptr;
        }

        if (c->bin_num == kInvalidBinNum) {
          InsertFreeChunkIntoBin(h);
        }
      }
      h = c->next;
    }
  }
}

// StreamAwareArena::StreamAwareArena(std::unique_ptr<OrtAllocator> allocator, const ArenaConfig& config,
//                                    bool enable_cross_stream_sharing,
//                                    const OrtApi& api, const OrtLogger& logger)
//     : ArenaImpl{ArenaType::StreamAwareArena, std::move(allocator), config, api, logger},
//       enable_cross_stream_usage_{enable_cross_stream_sharing} {
// }
//
// void* StreamAwareArena::AllocOnStream(size_t size, OrtSyncStream* current_stream, WaitNotificationFn wait_fn) {
//   return AllocateRawInternal(size, false, current_stream, enable_cross_stream_usage_, wait_fn);
// }
//
// void StreamAwareArena::ReleaseStreamBuffers(OrtSyncStream* stream) {
//   // since chunks on target stream will be reset to nullptr, trigger coalesce to see whether we can get bigger chunk.
//   ResetChunkOnTargetStream(stream, true);
// }

}  // namespace ep_utils
}  // namespace onnxruntime

// Need to figure out: Call to UpdateProducerStreamSyncInfo and GetCurrentSyncId.
