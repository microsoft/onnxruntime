#pragma once
#include "MemoryAllocator.h"

namespace Dml
{
    struct HeapMapping
    {
        ID3D12Heap *Heap;
        MemorySegment ResourceSegment;
        std::vector<MemorySegment> HeapSegments;

        bool operator==(const HeapMapping &) const;
    };

    using HeapMappings = std::vector<HeapMapping>;

    struct HeapMappingsComparer
    {
        bool operator()(const HeapMappings &a, const HeapMappings &b) const;
    };

    struct HeapMappingsHasher
    {
        size_t operator()(const HeapMappings &value) const;
    };

    struct ResourceComparer
    {
        bool operator()(const ComPtr<ID3D12Resource> &a, const ComPtr<ID3D12Resource> &b) const;
    };

    struct ResourceHasher
    {
        size_t operator()(const ComPtr<ID3D12Resource> &value) const;
    };
} // namespace Dml