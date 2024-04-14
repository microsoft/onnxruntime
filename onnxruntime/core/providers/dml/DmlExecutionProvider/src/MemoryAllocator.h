#pragma once
#include "precomp.h"

namespace Dml
{
    struct MemorySegment
    {
        uint64_t Start = 0, Size = 0;

        uint64_t End() const noexcept;

        bool OverlapsWith(const MemorySegment& other) const noexcept;

        explicit operator bool() const noexcept;

        bool operator==(const MemorySegment&) const noexcept;
        bool operator<(const MemorySegment&) const noexcept;
    };

    class MemoryAllocator
    {
    public:
        MemoryAllocator(uint64_t size = 0);

        void GrowBy(uint64_t size);
        void Reset(uint64_t size = 0);

        uint64_t FreeSpace() const;
        uint64_t UsedSpace() const;
        uint64_t Capacity() const;

        MemorySegment TryAllocate(uint64_t size);
        bool TryDeallocate(MemorySegment segment);

        std::vector<MemorySegment> TryMultipartAllocate(uint64_t size);

    private:
        uint64_t m_capacity;
        std::vector<MemorySegment> m_freeSpace;

        MemorySegment TryAllocateFrom(MemorySegment* segment, uint64_t size);

        MemorySegment* SmallestFreeSegment(uint64_t minSize = 0);
        MemorySegment* LargestFreeSegment();

        void AssertIntegrity() const;
        bool TestIntegrity() const;
    };
} // namespace Dml