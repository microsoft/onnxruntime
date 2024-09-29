#pragma once
#include "HeapMappings.h"

namespace Dml
{
    class HeapAllocator
    {
        struct HeapBlock
        {
            uint64_t Size;
            ComPtr<ID3D12Heap> Heap;
        };

        struct ResourceCache
        {
            ComPtr<ID3D12Resource> Resource;
            uint32_t Age = 0;
        };

    public:
        HeapAllocator(ID3D12Device* device, ID3D12CommandQueue* queue);

        ComPtr<ID3D12Resource> AllocateBuffer(uint64_t size);
        bool TryReleaseBuffer(const ComPtr<ID3D12Resource>& buffer);

        std::vector<ComPtr<IUnknown>> Clean();
        std::vector<ComPtr<IUnknown>> Clear();

    private:
        static const uint64_t m_blockSize;

        ComPtr<ID3D12Device> m_device;
        ComPtr<ID3D12CommandQueue> m_queue;

        std::vector<HeapBlock> m_heaps;
        MemoryAllocator m_allocator;
        uint64_t m_usedHeapSpace = 0, m_maxUsedHeapSpace = 0;
        bool m_clearRequested = false;

        std::unordered_map<ComPtr<ID3D12Resource>, HeapMappings, ResourceHasher, ResourceComparer> m_usedResources;
        std::unordered_map<HeapMappings, ResourceCache, HeapMappingsHasher, HeapMappingsComparer> m_freeResources;

        void EnsureHeapSpace(uint64_t size);
        ComPtr<ID3D12Resource> AllocateResource(uint64_t size);

        std::vector<HeapMapping> CalculateHeapMappings(gsl::span<MemorySegment> segments) const;
        ComPtr<ID3D12Resource> CreateResource(uint64_t size);
        void UpdateTileMappings(ID3D12Resource* resource, gsl::span<HeapMapping> mappings);
    };
} // namespace Dml