#include "precomp.h"

namespace Dml
{
  struct MemorySegment
  {
    uint64_t Start = 0, Size = 0;

    explicit operator bool() const noexcept;
  };

  uint64_t AlignMemoryOffset(uint64_t offset, uint64_t alignment);

  class MemoryAllocator
  {
  public:
    MemoryAllocator(uint64_t size);

    uint64_t FreeSpace() const;
    uint64_t UsedSpace() const;
    uint64_t Capacity() const;

    MemorySegment TryAllocate(uint64_t size, uint64_t alignment = 0);
    void Deallocate(MemorySegment segment);

    std::vector<MemorySegment> PartiallyAllocate(uint64_t size, uint64_t alignment = 0, uint64_t* remaining = nullptr);

  private:
    uint64_t m_capacity;
    std::vector<MemorySegment> m_freeSpace;

    MemorySegment TryAllocateFrom(MemorySegment* segment, uint64_t size, uint64_t alignment);
  };
}