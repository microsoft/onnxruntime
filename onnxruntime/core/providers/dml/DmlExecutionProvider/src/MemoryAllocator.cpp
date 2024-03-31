#include "precomp.h"
#include "MemoryAllocator.h"

using namespace std;

namespace Dml
{
  MemoryAllocator::MemoryAllocator(uint64_t size) :
    m_capacity(size)
  {
    m_freeSpace.push_back({ 0, m_capacity });
  }

  void MemoryAllocator::GrowBy(uint64_t size)
  {
    //The capacity will increase by the specified amount
    auto newCapacity = size + m_capacity;

    //If we have no free space or the last segment is not at the end, we need a new segment of free space
    if (m_freeSpace.empty() || m_freeSpace.back().End() != m_capacity)
    {
      m_freeSpace.push_back({ m_capacity, size });
    }
    //Otherwise we can make the last free segment longer
    else
    {
      m_freeSpace.back().Size += size;
    }

    //Set new capacity
    m_capacity = newCapacity;
  }

  uint64_t MemoryAllocator::FreeSpace() const
  {
    uint64_t result = 0;
    for (auto& segment : m_freeSpace)
    {
      result += segment.Size;
    }
    return result;
  }

  uint64_t MemoryAllocator::UsedSpace() const
  {
    return Capacity() - FreeSpace();
  }

  uint64_t MemoryAllocator::Capacity() const
  {
    return m_capacity;
  }

  MemorySegment MemoryAllocator::TryAllocate(uint64_t size)
  {
    //If we do not have empty space return or the capacity is too small return
    if (m_freeSpace.empty() || size > m_capacity) return {};

    //Otherwise find smallest empty slot
    auto segment = SmallestFreeSegment(size);
    return TryAllocateFrom(segment, size);
  }

  void MemoryAllocator::Deallocate(MemorySegment segment)
  {
    //Find neighboring slots
    MemorySegment* previousSlot = nullptr;
    MemorySegment* nextSlot = nullptr;

    auto segmentEnd = segment.Start + segment.Size;
    for (auto& slot : m_freeSpace)
    {
      if (segmentEnd == slot.Start) nextSlot = &slot;

      auto currentEnd = slot.Start + slot.Size;
      if (currentEnd == segment.Start) previousSlot = &slot;
    }

    //Add neighboring segments to current one
    if (previousSlot)
    {
      segment.Start = previousSlot->Start;
      segment.Size += previousSlot->Size;
      swap(m_freeSpace.back(), *previousSlot);
      m_freeSpace.pop_back();
    }

    if (nextSlot)
    {
      segment.Size += nextSlot->Size;
      swap(m_freeSpace.back(), *nextSlot);
      m_freeSpace.pop_back();
    }

    //Add reclaimed segment
    m_freeSpace.push_back(segment);
  }

  std::vector<MemorySegment> MemoryAllocator::TryMultipartAllocate(uint64_t size)
  {
    //Check if we have enough free space
    if (FreeSpace() < size) return {};

    //Try allocating in smallest continuous segment
    auto continuousAllocation = TryAllocate(size);
    if (continuousAllocation) return { continuousAllocation };

    //Try multipart allocation
    vector<MemorySegment> allocations;
    uint64_t remainingSize = size;
    while (remainingSize > 0)
    {
      auto segment = LargestFreeSegment();
      auto allocation = TryAllocateFrom(segment, min(segment->Size, remainingSize));
      remainingSize -= allocation.Size;
      allocations.push_back(allocation);
    }
    return allocations;
  }

  MemorySegment MemoryAllocator::TryAllocateFrom(MemorySegment* segment, uint64_t size)
  {
    //Validate inputs
    if (!segment || m_freeSpace.empty() || segment < &m_freeSpace.front() || segment > &m_freeSpace.back() || segment->Size < size) return {};

    //Decrease free space
    auto allocationStart = segment->Start;
    segment->Start = segment->Start + size;
    segment->Size -= size;

    //Remove empty slot if needed
    if (segment->Size == 0)
    {
      swap(m_freeSpace.back(), *segment);
      m_freeSpace.pop_back();
    }

    //Return allocated segment
    return { allocationStart, size };
  }

  MemorySegment * MemoryAllocator::SmallestFreeSegment(uint64_t minSize)
  {
    MemorySegment* selectedSlot = nullptr;
    for (auto& slot : m_freeSpace)
    {
      if (slot.Size >= minSize && (!selectedSlot || selectedSlot->Size > slot.Size))
      {
        selectedSlot = &slot;
      }
    }
    return selectedSlot;
  }

  MemorySegment* MemoryAllocator::LargestFreeSegment()
  {
    MemorySegment* selectedSlot = nullptr;
    for (auto& slot : m_freeSpace)
    {
      if (!selectedSlot || selectedSlot->Size < slot.Size)
      {
        selectedSlot = &slot;
      }
    }
    return selectedSlot;
  }

  uint64_t MemorySegment::End() const noexcept
  {
    return Start + Size;
  }

  MemorySegment::operator bool() const noexcept
  {
    return Size > 0u;
  }

  bool MemorySegment::operator==(const MemorySegment& other) const noexcept
  {
    return Start == other.Start && Size == other.Size;
  }

  uint64_t AlignMemoryOffset(uint64_t offset, uint64_t alignment)
  {
    auto alignmentRemainder = offset % alignment;
    if (alignmentRemainder != 0) offset += alignment - alignmentRemainder;
    return offset;
  }
}