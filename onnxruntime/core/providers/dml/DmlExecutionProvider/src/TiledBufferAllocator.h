#pragma once
#include "ExecutionContext.h"
#include "ExecutionProvider.h"
#include "DmlResourceWrapper.h"
#include "HeapAllocator.h"
#include "DmlBufferAllocator.h"

namespace Dml
{
  class DmlSubAllocator;

  class TiledBufferAllocator : public DmlBufferAllocator
  {
    struct BackingHeap
    {
      ComPtr<ID3D12Heap> Heap;
    };

  public:
    TiledBufferAllocator(
      ID3D12Device* device,
      std::shared_ptr<ExecutionContext> context,
      const D3D12_HEAP_PROPERTIES& heapProps,
      D3D12_HEAP_FLAGS heapFlags,
      D3D12_RESOURCE_FLAGS resourceFlags,
      D3D12_RESOURCE_STATES initialState,
      std::unique_ptr<DmlSubAllocator>&& subAllocator);
        
    virtual void SetResidency(bool value) override;

    using DmlBufferAllocator::Alloc;
    virtual void* Alloc(size_t size, AllocatorRoundingMode roundingMode) override;

  private:
    ComPtr<ID3D12Device> m_device;
    D3D12_HEAP_PROPERTIES m_heapProperties;
    D3D12_HEAP_FLAGS m_heapFlags;
    D3D12_RESOURCE_FLAGS m_resourceFlags;
    D3D12_RESOURCE_STATES m_initialState;    
    std::shared_ptr<ExecutionContext> m_context;
    std::unique_ptr<DmlSubAllocator> m_subAllocator;

    HeapAllocator m_heapAllocator, m_longAllocator;
    uint64_t m_currentAllocationId = 0, m_currentResourceId = 0;
    uint64_t m_tiledAllocationSize = 0, m_untiledAllocationSize = 0;
    std::unordered_map<uintptr_t, uint64_t> m_resourceIds;

    virtual void FreeResource(void* p, uint64_t resourceId) override;
  };
}