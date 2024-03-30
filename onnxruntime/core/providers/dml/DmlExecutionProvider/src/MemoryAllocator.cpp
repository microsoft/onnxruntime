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

  MemorySegment MemoryAllocator::TryAllocate(uint64_t size, uint64_t alignment)
  {
    //If we do not have empty space return or the capacity is too small return
    if (m_freeSpace.empty() || size > m_capacity) return {};

    //Otherwise find smallest empty slot
    MemorySegment* selectedSlot = nullptr;
    for (auto& slot : m_freeSpace)
    {
      //If slot is smaller than the alignment ignore it
      if (slot.Size < alignment) continue;

      //Calculate space after alignment
      auto alignedStart = AlignMemoryOffset(slot.Start, alignment);
      auto remainingSpace = slot.Start + slot.Size - alignedStart;

      //If we have space then select the slot
      if (remainingSpace >= size && (!selectedSlot || selectedSlot->Size > slot.Size))
      {
        selectedSlot = &slot;
      }
    }

    //Allocate segment
    return TryAllocateFrom(selectedSlot, size, alignment);
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

  vector<MemorySegment> MemoryAllocator::PartiallyAllocate(uint64_t size, uint64_t alignment, uint64_t * remaining)
  {
    auto allocation = TryAllocate(size, alignment);
    
    return vector<MemorySegment>();
  }

  MemorySegment MemoryAllocator::TryAllocateFrom(MemorySegment* segment, uint64_t size, uint64_t alignment)
  {
    //Validate inputs
    if (!segment || m_freeSpace.empty() || segment < &m_freeSpace.front() || segment > &m_freeSpace.back()) return {};

    auto alignedStart = AlignMemoryOffset(segment->Start, alignment);
    if (alignedStart + size > segment->Size) return {};

    //Decrease free space
    auto alignmentGap = alignedStart - segment->Start;
    segment->Start = alignedStart + size;
    segment->Size -= size + alignmentGap;

    //Remove empty slot if needed
    if (segment->Size == 0)
    {
      swap(m_freeSpace.back(), *segment);
      m_freeSpace.pop_back();
    }

    //Keep alignment gap as free space
    if (alignmentGap > 0u)
    {
      m_freeSpace.push_back(MemorySegment{alignedStart - alignmentGap, alignmentGap});
    }

    //Return allocated segment
    return { alignedStart, size };
  }

  MemorySegment::operator bool() const noexcept
  {
    return Size > 0u;
  }

  uint64_t AlignMemoryOffset(uint64_t offset, uint64_t alignment)
  {
    auto alignmentRemainder = offset % alignment;
    if (alignmentRemainder != 0) offset += alignment - alignmentRemainder;
    return offset;
  }
}