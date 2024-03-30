#pragma once
#include "core/framework/allocator.h"
#include "ExecutionContext.h"
#include "ExecutionProvider.h"
#include "DmlResourceWrapper.h"
#include "AllocationInfo.h"

namespace Dml
{
  class DmlSubAllocator;

  class TiledBufferAllocator : public onnxruntime::IAllocator
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
        
    const AllocationInfo* DecodeDataHandle(const void* opaqueHandle);

    void SetDefaultRoundingMode(AllocatorRoundingMode roundingMode);
        
    void SetResidency(bool value);

    void* Alloc(size_t size) final;
    void* Alloc(size_t size, AllocatorRoundingMode roundingMode);
    void Free(void* p) final;

  private:
    ComPtr<ID3D12Device> m_device;
    D3D12_HEAP_PROPERTIES m_heapProperties;
    D3D12_HEAP_FLAGS m_heapFlags;
    D3D12_RESOURCE_FLAGS m_resourceFlags;
    D3D12_RESOURCE_STATES m_initialState;    
    std::shared_ptr<ExecutionContext> m_context;
    std::unique_ptr<DmlSubAllocator> m_subAllocator;

    AllocatorRoundingMode m_defaultRoundingMode = AllocatorRoundingMode::Disabled;


  };
}