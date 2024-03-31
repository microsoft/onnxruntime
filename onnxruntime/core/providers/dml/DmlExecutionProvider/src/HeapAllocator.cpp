#include "precomp.h"
#include "HeapAllocator.h"

using namespace Dml;

namespace {
  uint64_t RoundToBlockSize(uint64_t allocationSize, uint64_t blockSize)
  {
    auto blockRemainder = allocationSize % blockSize;
    if (blockRemainder != 0) allocationSize += blockSize - blockRemainder;
    return allocationSize;
  }
}

namespace Dml
{
  const uint64_t HeapAllocator::m_blockSize = 64 * 1024 * 1024;

  HeapAllocator::HeapAllocator(ID3D12Device * device, ID3D12CommandQueue * queue) :
    m_device(device),
    m_queue(queue)
  { }

  ComPtr<ID3D12Resource> HeapAllocator::AllocateBuffer(uint64_t requestedSize)
  {
    //Round up to tile size
    auto allocationSize = RoundToBlockSize(requestedSize, D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES);

    //Ensure heap space
    EnsureHeapSpace(allocationSize);

    //Allocate resource
    return AllocateResource(allocationSize);
  }

  void HeapAllocator::ReleaseBuffer(ComPtr<ID3D12Resource>&& buffer)
  {
    //Unregister used resource
    auto heapMappings{ m_usedResources.extract(buffer).mapped() };
    
    //Deallocate all mappings
    uint64_t heapOffset = 0;
    auto heap = m_heaps.begin();
    for (auto& mapping : heapMappings)
    {
      //Find current heap
      while(heap->Heap.Get() != mapping.Heap)
      {
        heapOffset += heap->Size;
        heap++;
      }

      //Deallocate all segments in heap
      for (auto& segment : mapping.HeapSegments)
      {
        m_allocator.Deallocate({ segment.Start + heapOffset, segment.Size });
      }
    }
    
    //Register as free resource
    m_freeResources[std::move(heapMappings)] = std::move(buffer);    
  }

  void HeapAllocator::EnsureHeapSpace(uint64_t size)
  {
    //Check if we have enough free space
    auto freeSpace = m_allocator.FreeSpace();
    if (freeSpace >= size) return;

    //If not calculate the additional space needed
    //Round up the required space to the next block - to prevent many small heaps
    auto requiredSpace = RoundToBlockSize(size - freeSpace, m_blockSize);

    //Allocate new heap
    ComPtr<ID3D12Heap> heap;
    CD3DX12_HEAP_DESC heapDescription{requiredSpace, D3D12_HEAP_TYPE_DEFAULT};
    ORT_THROW_IF_FAILED(m_device->CreateHeap(
      &heapDescription, 
      IID_PPV_ARGS(heap.GetAddressOf()))
    );

    //Register heap
    m_allocator.GrowBy(requiredSpace);
    m_heaps.push_back({ requiredSpace, std::move(heap) });
  }

  ComPtr<ID3D12Resource> HeapAllocator::AllocateResource(uint64_t size)
  {
    //Register allocation
    auto segments = m_allocator.TryMultipartAllocate(size);

    //Map segments to heaps
    auto heapMappings = CalculateHeapMappings(segments);

    //Check if we need a new resource
    auto it = m_freeResources.find(heapMappings);
    if (it == m_freeResources.end())
    {
      auto resource = CreateResource(size);
      UpdateTileMappings(resource.Get(), heapMappings);
      
      m_usedResources[resource] = heapMappings;
      return resource;
    }
    //Or can reuse an existing one
    else
    {
      auto keyValuePair = m_freeResources.extract(it);
      m_usedResources[keyValuePair.mapped()] = keyValuePair.key();
      return keyValuePair.mapped();
    }
  }

  std::vector<HeapMapping> HeapAllocator::CalculateHeapMappings(gsl::span<MemorySegment> segments) const
  {
    std::vector<HeapMapping> results;

    auto heap = m_heaps.begin();
    auto segment = segments.begin();
    uint64_t heapOffset = 0;
    uint64_t resourceOffset = 0;
    while (segment != segments.end())
    {
      //Search for the heap containing the current segment start
      while (segment->Start > heapOffset + heap->Size)
      {
        heapOffset += heap++->Size;
      }
            
      //Define mapping in heap
      MemorySegment heapSegment;
      heapSegment.Start = segment->Start - heapOffset;
      heapSegment.Size = std::min(segment->End() - heapOffset, heap->Size) - heapSegment.Start;

      //Create new or get current resource mapping
      if (results.empty() || results.back().Heap != heap->Heap.Get())
      {
        results.push_back(HeapMapping{ heap->Heap.Get(), MemorySegment{ resourceOffset, 0 } });
      }
      auto& mapping = results.back();

      //Extend mapping
      mapping.HeapSegments.push_back(heapSegment);
      mapping.ResourceSegment.Size += heapSegment.Size;
      resourceOffset += heapSegment.Size;

      //Advance segment if we reached its end
      if (segment->End() - heapOffset >= heapOffset + heap->Size)
      {
        segment++;
      }
      //Alternatively advance the heap
      else
      {
        heapOffset += heap++->Size;
      }
    }

    return results;
  }

  ComPtr<ID3D12Resource> HeapAllocator::CreateResource(uint64_t size)
  {
    ComPtr<ID3D12Resource> resource;
    auto resourceDescription = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ORT_THROW_IF_FAILED(m_device->CreateReservedResource(
      &resourceDescription,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(resource.GetAddressOf()))
    );
    return resource;
  }

  void HeapAllocator::UpdateTileMappings(ID3D12Resource* resource, gsl::span<HeapMapping> mappings)
  {
    for (auto& mapping : mappings)
    {
      //Resource mapping
      D3D12_TILED_RESOURCE_COORDINATE resourceStart{ UINT(mapping.ResourceSegment.Start / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES), 0, 0, 0 };

      D3D12_TILE_REGION_SIZE resourceSize{ UINT(mapping.ResourceSegment.Size / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES), false, 0, 0, 0 };

      //Heap mappings
      std::vector<D3D12_TILE_RANGE_FLAGS> heapFlags; heapFlags.reserve(mapping.HeapSegments.size());
      std::vector<UINT> heapStarts; heapStarts.reserve(mapping.HeapSegments.size());
      std::vector<UINT> heapCounts; heapCounts.reserve(mapping.HeapSegments.size());
      for (auto& heapSegment : mapping.HeapSegments)
      {
        heapFlags.push_back(D3D12_TILE_RANGE_FLAG_NONE);
        heapStarts.push_back(UINT(heapSegment.Start / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES));
        heapCounts.push_back(UINT(heapSegment.Size / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES));
      }

      //Upload tile mappings
      m_queue->UpdateTileMappings(
        resource,
        1,
        &resourceStart,
        &resourceSize,
        mapping.Heap,
        UINT(mapping.HeapSegments.size()),
        heapFlags.data(),
        heapStarts.data(),
        heapCounts.data(),
        D3D12_TILE_MAPPING_FLAG_NONE);
    }
  }
};