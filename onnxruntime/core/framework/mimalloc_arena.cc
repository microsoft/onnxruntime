#if defined(USE_MIMALLOC_ARENA_ALLOCATOR)
#include "mimalloc.h"
#include "core/framework/mimalloc_arena.h"

namespace onnxruntime {
MiMallocArena::MiMallocArena(std::unique_ptr<IAllocator> resource_allocator,
                             size_t total_memory)
    : info_(resource_allocator->Info().name, OrtAllocatorType::OrtArenaAllocator, resource_allocator->Info().device, resource_allocator->Info().id, resource_allocator->Info().mem_type) {
  stats_.bytes_limit = total_memory;
}

void* MiMallocArena::Alloc(size_t size) {
#if (MI_STAT > 1)
  stats_.num_allocs++;
#endif
  return mi_malloc(size);
}

void MiMallocArena::Free(void* p) {
  mi_free(p);
}

void* MiMallocArena::Reserve(size_t size) {
  return mi_malloc(size);
}

// mimalloc only maintains stats when compiled under debug (which in turn sets MI_STAT)
void MiMallocArena::GetStats(AllocatorStats* stats) {
#if (MI_STAT > 1)
  auto current_stats = mi_heap_get_default()->tld->stats;
  stats_.bytes_in_use = current_stats.malloc.current;
  stats_.total_allocated_bytes = current_stats.reserved.current;
  stats_.max_bytes_in_use = current_stats.reserved.peak;
#endif
  *stats = stats_;
}

size_t MiMallocArena::Used() const {
#if (MI_STAT > 1)
  return mi_heap_get_default()->tld->stats.malloc.current;
#else
  return 0;
#endif
}

size_t MiMallocArena::AllocatedSize(const void* ptr) {
  return mi_usable_size(ptr);
}
}  // namespace onnxruntime
#endif
