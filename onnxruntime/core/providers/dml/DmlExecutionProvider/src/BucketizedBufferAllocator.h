// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ExecutionContext.h"
#include "DmlResourceWrapper.h"
#include "DmlBufferAllocator.h"

namespace Dml
{
    class DmlSubAllocator;
    
    // Implements a Lotus allocator for D3D12 heap buffers, using a bucket allocation strategy. The allocator
    // maintains a set of fixed-size buckets, with each bucket containing one or more D3D12 buffers of that fixed size.
    // All requested allocation sizes are rounded up to the nearest bucket size, which ensures minimal fragmentation
    // while providing an upper bound on the amount of memory "wasted" with each allocation.
    class BucketizedBufferAllocator : public DmlBufferAllocator
    {
    public:
        ~BucketizedBufferAllocator();

        // Constructs a BucketizedBufferAllocator which allocates D3D12 committed resources with the specified heap properties,
        // resource flags, and initial resource state.
        BucketizedBufferAllocator(
            ID3D12Device* device,
            ExecutionContext* context,
            const D3D12_HEAP_PROPERTIES& heapProps,
            D3D12_HEAP_FLAGS heapFlags,
            D3D12_RESOURCE_FLAGS resourceFlags,
            D3D12_RESOURCE_STATES initialState,
            std::unique_ptr<DmlSubAllocator>&& subAllocator);

    public: // onnxruntime::IAllocator
        using DmlBufferAllocator::Alloc;
        virtual void* Alloc(size_t size, AllocatorPoolingMode poolingMode) override;

        virtual DmlAllocatorType Type() const override;

    private:
        static const uint32_t c_minResourceSizeExponent = 16; // 2^16 = 64KB

        // The pool consists of a number of buckets, and each bucket contains a number of resources of the same size.
        // The resources in each bucket are always sized as a power of two, and each bucket contains resources twice
        // as large as the previous bucket.
        struct Resource
        {
            ComPtr<DmlResourceWrapper> resource;
            uint64_t resourceId;
        };

        struct Bucket
        {
            std::vector<Resource> resources;
        };

        static gsl::index GetBucketIndexFromSize(uint64_t size);
        static uint64_t GetBucketSizeFromIndex(gsl::index index);

        virtual void FreeResource(void* p, uint64_t resourceId) override;

        ComPtr<ID3D12Device> m_device;
        D3D12_HEAP_PROPERTIES m_heapProperties;
        D3D12_HEAP_FLAGS m_heapFlags;
        D3D12_RESOURCE_FLAGS m_resourceFlags;
        D3D12_RESOURCE_STATES m_initialState;

        std::vector<Bucket> m_pool;
        size_t m_currentAllocationId = 0;
        uint64_t m_currentResourceId = 0;

        ComPtr<ExecutionContext> m_context;
        std::unique_ptr<DmlSubAllocator> m_subAllocator;

    #ifndef NDEBUG
        // Useful for debugging; keeps track of all allocations that haven't been freed yet
        std::map<size_t, AllocationInfo*> m_outstandingAllocationsById;
    #endif
    };

} // namespace Dml
