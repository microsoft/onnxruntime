#pragma once
#include "DmlBufferAllocator.h"
#include "DmlResourceWrapper.h"
#include "ExecutionContext.h"
#include "ExecutionProvider.h"
#include "HeapAllocator.h"

#define DML_USE_HEAP_ALLOCATOR_FOR_UNPOOLED_MEMORY

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
        TiledBufferAllocator(ID3D12Device* device, ExecutionContext* context,
                             const D3D12_HEAP_PROPERTIES& heapProps, D3D12_HEAP_FLAGS heapFlags,
                             D3D12_RESOURCE_FLAGS resourceFlags, D3D12_RESOURCE_STATES initialState,
                             std::unique_ptr<DmlSubAllocator>&& subAllocator);

        using DmlBufferAllocator::Alloc;
        virtual void* Alloc(size_t size, AllocatorPoolingMode poolingMode) override;

        virtual void Clean() override;
        void Clear();

        virtual DmlAllocatorType Type() const override;

    protected:
        virtual void FreeResource(void* p, uint64_t resourceId) override;

    private:
        ComPtr<ID3D12Device> m_device;
        D3D12_HEAP_PROPERTIES m_heapProperties;
        D3D12_HEAP_FLAGS m_heapFlags;
        D3D12_RESOURCE_FLAGS m_resourceFlags;
        D3D12_RESOURCE_STATES m_initialState;
        ExecutionContext* m_context;

        HeapAllocator m_pooledAllocator;

#ifdef DML_USE_HEAP_ALLOCATOR_FOR_UNPOOLED_MEMORY
        HeapAllocator m_unpooledAllocator;
#else
        std::unique_ptr<DmlSubAllocator> m_unpooledAllocator;
#endif

        uint64_t m_currentAllocationId = 0, m_currentResourceId = 0;
        std::unordered_map<uintptr_t, uint64_t> m_resourceIds;

        void FreeResources(std::vector<Microsoft::WRL::ComPtr<IUnknown>>& resources);
    };
} // namespace Dml