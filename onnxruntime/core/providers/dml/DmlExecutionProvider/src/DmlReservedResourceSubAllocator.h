// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ExecutionContext.h"
#include "DmlAllocationInfo.h"
#include "DmlBufferRegion.h"
#include "DmlSubAllocator.h"

namespace Dml
{
    // An allocator that makes logically contiguous allocations backed by D3D heaps.
    //
    // Heaps must fit entirely in either local or non-local memory. Larger heaps
    // have a greater chance of getting demoted into non-local memory, which can be
    // disastrous for performance. This problem is compounded by the fact that heaps
    // may be demoted even if overall local memory usage is within the process'
    // budget. Heaps are not necessarily mappable to discontiguous regions of
    // physical memory, which means physical memory fragmentation *may* make it
    // extremely difficult to accommodate larger heaps.
    //
    // On D3D hardware that supports tiled resource tier 1+ this class implements
    // large allocations through tiling. Each allocation is backed by however many
    // small heaps are necessary to cover the requested allocation size. Buffer
    // regions retrieved through this allocator are reserved resources that span the
    // full collection of heaps assigned to an individual allocation. Tile mappings
    // are static.
    //
    // On hardware that doesn't support tiled resources each allocation is backed by
    // a single heap. Buffer regions retrieved through this allocator are placed
    // resources that span the full heap assigned to an individual allocation. In
    // this case it is better make more but smaller allocations (resulting in
    // smaller heaps); this fallback path is only retained as a last resort for
    // older hardware.
    class DmlReservedResourceSubAllocator : public DmlSubAllocator
    {
    public:
        // Maximum size of a heap (in tiles) when allocations are tiled. Each tile
        // is 64KB. A default size of 512 tiles (32MB) does a good job of handling
        // local video memory fragmentation without requiring lots of heaps.
        static constexpr uint64_t DefaultMaxHeapSizeInTiles = 512;

        DmlReservedResourceSubAllocator(
            ID3D12Device* device,
            std::shared_ptr<ExecutionContext> context,
            ID3D12CommandQueue* queue,
            const D3D12_HEAP_PROPERTIES& heapProps,
            D3D12_HEAP_FLAGS heapFlags,
            D3D12_RESOURCE_FLAGS resourceFlags,
            D3D12_RESOURCE_STATES initialState);

        // Creates a reserved or placed resource buffer over the given memory range.
        // The physical D3D12 resource may be larger than the requested size, so
        // callers must ensure to use the offset/size returned in the
        // D3D12BufferRegion else risk out of bounds access. Note that in practice
        // the ID3D12Resource is cached, so this call typically has a lower cost
        // than a call to ID3D12Device::CreatePlacedResource or
        // CreateReservedResource.
        D3D12BufferRegion CreateBufferRegion(void* opaquePointer, uint64_t sizeInBytes);

        AllocationInfo* GetAllocationInfo(void* opaquePointer);

        void FreeResource(AllocationInfo* allocInfo, uint64_t resourceId) final;
        uint64_t ComputeRequiredSize(size_t size);
        bool TilingEnabled() const { return m_tilingEnabled; };
        uint64_t GetUniqueId(void* opaquePointer);

        ~DmlReservedResourceSubAllocator();

        // Constructs a DmlReservedResourceSubAllocator which allocates D3D12 committed resources with the specified heap properties,
        // resource flags, and initial resource state.
        DmlReservedResourceSubAllocator(
            ID3D12Device* device,
            std::shared_ptr<ExecutionContext> context,
            std::unique_ptr<DmlReservedResourceSubAllocator>&& subAllocator);

        void* Alloc(size_t size);
        void Free(void* p);

    private:
        static constexpr uint32_t MinResourceSizeExponent = 16; // 2^16 = 64KB

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

        friend class AllocationInfo;

        std::vector<Bucket> m_pool;
        size_t m_currentUniqueAllocationId = 0;
        uint64_t m_currentResourceId = 0;
        std::unique_ptr<DmlReservedResourceSubAllocator> m_subAllocator;

    #if _DEBUG
        // Useful for debugging; keeps track of all allocations that haven't been freed yet
        std::map<size_t, AllocationInfo*> m_outstandingAllocationsById;
    #endif

        std::mutex m_mutex;

        Microsoft::WRL::ComPtr<ID3D12Device> m_device;
        std::shared_ptr<ExecutionContext> m_context;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_queue;
        const D3D12_HEAP_PROPERTIES m_heapProperties;
        const D3D12_HEAP_FLAGS m_heapFlags;
        const D3D12_RESOURCE_FLAGS m_resourceFlags;
        const D3D12_RESOURCE_STATES m_initialState;
        bool m_tilingEnabled;
        uint64_t m_maxHeapSizeInTiles;

        // The largest allocation ID we've returned so far (or 0 if we've never done
        // so). Note that our allocation IDs start at 1 (not 0) to ensure that it
        // isn't possible for a valid allocation to have a pointer value of
        // 0x00000000.
        uint32_t m_currentAllocationId = 0;

        // A list of unused allocation IDs. This is for re-use of IDs once they get
        // freed. We only bump the max_allocation_id_ once there are no more free
        // IDs.
        std::vector<uint32_t> m_freeAllocationIds;

        absl::optional<DmlHeapAllocation> TryCreateTiledAllocation(uint64_t sizeInBytes);
        absl::optional<DmlHeapAllocation> TryCreateUntiledAllocation(uint64_t sizeInBytes);

        friend class D3D12BufferRegion;

        absl::flat_hash_map<uint32_t, Microsoft::WRL::ComPtr<AllocationInfo>> m_allocationsById;

        // Retrieves a free allocation ID, or nullopt if no more IDs are available.
        absl::optional<uint32_t> TryReserveAllocationID();

        // Releases an allocation ID back to the pool of IDs.
        void ReleaseAllocationID(uint32_t id);
    };

} // namespace Dml
