#if defined(USE_MIMALLOC_ARENA_ALLOCATOR)
#include "core/common/common.h"
#include "core/framework/arena.h"
#include "onnxruntime_config.h"

namespace onnxruntime {
class MiMallocArena : public IArenaAllocator {
 public:
  MiMallocArena(std::unique_ptr<IAllocator> resource_allocator, size_t total_memory);

  void* Alloc(size_t size) override;

  void Free(void* p) override;

  // mimalloc only maintains stats when compiled under debug, or when MI_STAT >= 2
  void GetStats(AllocatorStats* stats);

  void* Reserve(size_t size) override;

  size_t Used() const override;

  size_t Max() const override {
    return stats_.bytes_limit;
  }

  const OrtMemoryInfo& Info() const override {
    return info_;
  }

  size_t AllocatedSize(const void* ptr);

  OrtMemoryInfo info_;
  AllocatorStats stats_;
};
}  // namespace onnxruntime
#endif
