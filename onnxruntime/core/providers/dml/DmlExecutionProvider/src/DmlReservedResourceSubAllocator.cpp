// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "core/session/onnxruntime_c_api.h"
#include "DmlReservedResourceSubAllocator.h"
#include "DmlReservedResourceWrapper.h"
#include "DmlBufferRegion.h"

namespace Dml
{
    DmlReservedResourceSubAllocator::~DmlReservedResourceSubAllocator()
    {
#ifdef PRINT_OUTSTANDING_ALLOCATIONS
        if (!m_outstandingAllocationsById.empty())
        {
            printf("DmlReservedResourceSubAllocator outstanding allocation indices:\n");
            for (auto& entry : m_outstandingAllocationsById)
            {
                printf("%u\n", static_cast<int>(entry.first));
            }
            printf("\n");
        }
#endif
    }

    /*static*/ gsl::index DmlReservedResourceSubAllocator::GetBucketIndexFromSize(uint64_t size)
    {
        assert(size != 0);

        // Each bucket is twice as large as the previous one, in ascending order
        gsl::index index = static_cast<gsl::index>(ceil(log2(size)));
        assert((1ull << index) >= size); // This must be true unless there were some strange rounding issues

        // The smallest bucket is 2^n bytes large, where n = MinResourceSizeExponent
        index = std::max<gsl::index>(index, MinResourceSizeExponent);
        index -= MinResourceSizeExponent;

        return index;
    }

    /*static*/ uint64_t DmlReservedResourceSubAllocator::GetBucketSizeFromIndex(gsl::index index)
    {
        return (1ull << (index + MinResourceSizeExponent));
    }

    static bool GetTilingEnabled(ID3D12Device* device)
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
        if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))))
        {
            return options.TiledResourcesTier >= D3D12_TILED_RESOURCES_TIER_1;
        }

        return false;
    }

    static uint64_t GetMaxHeapSizeInTiles()
    {
        return DmlReservedResourceSubAllocator::DefaultMaxHeapSizeInTiles;
    }

    DmlReservedResourceSubAllocator::DmlReservedResourceSubAllocator(
        ID3D12Device* device,
        std::shared_ptr<ExecutionContext> context,
        ID3D12CommandQueue* queue,
        const D3D12_HEAP_PROPERTIES& heapProps,
        D3D12_HEAP_FLAGS heapFlags,
        D3D12_RESOURCE_FLAGS resourceFlags,
        D3D12_RESOURCE_STATES initialState)
        : m_device(device),
        m_context(context),
        m_queue(queue),
        m_heapProperties(heapProps),
        m_heapFlags(heapFlags),
        m_resourceFlags(resourceFlags),
        m_initialState(initialState),
        m_tilingEnabled(GetTilingEnabled(device)),
        m_maxHeapSizeInTiles(GetMaxHeapSizeInTiles())
    {
    }

    absl::optional<DmlHeapAllocation> DmlReservedResourceSubAllocator::TryCreateTiledAllocation(uint64_t sizeInBytes)
    {
        DmlHeapAllocation allocation = {};

        // The allocation may be larger than the requested size to ensure a whole
        // number of tiles.
        const uint64_t resourceSizeInTiles = 1 + (sizeInBytes - 1) / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
        const uint64_t resourceSizeInBytes = resourceSizeInTiles * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
        auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(resourceSizeInBytes, m_resourceFlags);

        HRESULT createResourceHr = m_device->CreateReservedResource(
            &resourceDesc,
            m_initialState,
            nullptr,
            IID_PPV_ARGS(&allocation.resourceUavState));

        if (createResourceHr == E_OUTOFMEMORY)
        {
            return absl::nullopt;
        }
        ORT_THROW_IF_FAILED(createResourceHr);

        // Reserve enough heaps to store all tiles in the resource.
        const uint64_t heapCount = 1 + (resourceSizeInTiles - 1) / m_maxHeapSizeInTiles;
        allocation.heaps.resize(heapCount);

        // Create heaps and map them to the primary reserved resource.
        D3D12_TILED_RESOURCE_COORDINATE resourceRegionStartCoordinates = {};
        uint64_t unmappedResourceTiles = resourceSizeInTiles;
        for (uint64_t i = 0; i < heapCount; i++)
        {
            // Create heap. The last heap of the allocation may have fewer tiles to
            // avoid wasting space.
            uint64_t heapSizeInTiles = std::min<uint64_t>(unmappedResourceTiles, m_maxHeapSizeInTiles);
            uint64_t heapSizeInBytes = heapSizeInTiles * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
            auto heap_desc = CD3DX12_HEAP_DESC(
                heapSizeInBytes,
                m_heapProperties,
                0,
                m_heapFlags);

            HRESULT createHeapHr = m_device->CreateHeap(&heap_desc, IID_PPV_ARGS(&allocation.heaps[i]));
            if (createHeapHr == E_OUTOFMEMORY)
            {
                return absl::nullopt;
            }
            ORT_THROW_IF_FAILED(createHeapHr);

            // Source region in the resource to map.
            D3D12_TILE_REGION_SIZE resourceRegionSize = {};
            resourceRegionSize.NumTiles = static_cast<uint32_t>(heapSizeInTiles);

            // Target range in the current heap to map.
            constexpr D3D12_TILE_RANGE_FLAGS tileRangeFlags = D3D12_TILE_RANGE_FLAG_NONE;
            const uint32_t heapRangeTileCount = static_cast<uint32_t>(heapSizeInTiles);

            constexpr uint32_t heapRangeStartOffset = 0;
            constexpr uint32_t numResourceRegions = 1;
            constexpr uint32_t numHeapRanges = 1;

            // This is a brand new allocation/resource, so the tile mappings are
            // guaranteed to be set (on the GPU timeline) by the time any code can
            // reference the returned resource. We only execute operations on a
            // single hardware queue so there is no need to wait or signal.
            m_queue->UpdateTileMappings(
                allocation.resourceUavState.Get(),
                numResourceRegions,
                &resourceRegionStartCoordinates,
                &resourceRegionSize,
                allocation.heaps[i].Get(),
                numHeapRanges,
                &tileRangeFlags,
                &heapRangeStartOffset,
                &heapRangeTileCount,
                D3D12_TILE_MAPPING_FLAG_NONE);

            resourceRegionStartCoordinates.X += static_cast<uint32_t>(heapSizeInTiles);
            unmappedResourceTiles -= heapSizeInTiles;
        }

        assert(unmappedResourceTiles == 0);

        return allocation;
    }

    absl::optional<DmlHeapAllocation> DmlReservedResourceSubAllocator::TryCreateUntiledAllocation(uint64_t sizeInBytes)
    {
        DmlHeapAllocation allocation = {};

        // Create the allocation's sole heap. The allocation may be larger than the
        // requested size to ensure a whole number of tiles.
        allocation.heaps.resize(1);
        D3D12_HEAP_DESC heap_desc = CD3DX12_HEAP_DESC(sizeInBytes, m_heapProperties, 0, m_heapFlags);
        HRESULT createHeapHr = m_device->CreateHeap(&heap_desc, IID_PPV_ARGS(&allocation.heaps.front()));
        if (createHeapHr == E_OUTOFMEMORY)
        {
            return absl::nullopt;
        }
        ORT_THROW_IF_FAILED(createHeapHr);

        // Create large placed resource that spans the heap.
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, m_resourceFlags);

        HRESULT createResourceHr = m_device->CreatePlacedResource(
            allocation.heaps.front().Get(),
            0,
            &resourceDesc,
            m_initialState,
            nullptr,
            IID_PPV_ARGS(&allocation.resourceUavState));
        if (createResourceHr == E_OUTOFMEMORY)
        {
            return absl::nullopt;
        }
        ORT_THROW_IF_FAILED(createResourceHr);

        return allocation;
    }

    uint64_t DmlReservedResourceSubAllocator::ComputeRequiredSize(size_t size)
    {
        const uint64_t resourceSizeInTiles = 1 + (size - 1) / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
        const uint64_t resourceSizeInBytes = resourceSizeInTiles * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
        return resourceSizeInBytes;
    }

    void* DmlReservedResourceSubAllocator::Alloc(size_t sizeInBytes)
    {
        // For some reason lotus likes requesting 0 bytes of memory
        sizeInBytes = std::max<size_t>(1, sizeInBytes);

        // The D3D12 device is thread-safe so we don't need to hold the lock while
        // creating an allocation.
        absl::optional<DmlHeapAllocation> allocation =
            m_tilingEnabled ? TryCreateTiledAllocation(sizeInBytes)
                            : TryCreateUntiledAllocation(sizeInBytes);

        ORT_THROW_HR_IF(E_INVALIDARG, !allocation);

        // We need to access (mutable) state after this point, so we need to lock
        std::unique_lock<std::mutex> lock(m_mutex);

        absl::optional<uint32_t> allocationId = TryReserveAllocationID();
        ORT_THROW_HR_IF(E_INVALIDARG, !allocationId);

        auto resourceWrapper = wil::MakeOrThrow<DmlReservedResourceWrapper>(std::move(*allocation));
        ComPtr<AllocationInfo> allocInfo = wil::MakeOrThrow<AllocationInfo>(
            this,
            ++m_currentUniqueAllocationId,
            0,
            resourceWrapper.Get(),
            sizeInBytes
        );

        m_allocationsById.emplace(*allocationId, allocInfo);

        lock.unlock();

    #if _DEBUG
        m_outstandingAllocationsById[allocInfo->GetId()] = allocInfo.Get();
    #endif

        // DML only has a single device in ORT at the moment
        constexpr uint64_t deviceId = 0;
        constexpr uint64_t offset = 0;
        return TaggedPointer::Pack(deviceId, *allocationId, offset);
    }

    void DmlReservedResourceSubAllocator::Free(void* ptr)
    {
        ORT_THROW_HR_IF(E_INVALIDARG, ptr == nullptr);

        TaggedPointer taggedPtr = TaggedPointer::Unpack(ptr);
        ORT_THROW_HR_IF(E_INVALIDARG, taggedPtr.offset != 0);

        // We need to access (mutable) state after this point, so we need to lock
        std::unique_lock<std::mutex> lock(m_mutex);

        auto it = m_allocationsById.find(taggedPtr.allocationId);
        ORT_THROW_HR_IF(E_INVALIDARG, it == m_allocationsById.end());

        ReleaseAllocationID(taggedPtr.allocationId);

        // Frees the ID3D12Heap
        m_allocationsById.erase(it);
    }

    uint64_t DmlReservedResourceSubAllocator::GetUniqueId(void* opaquePointer)
    {
        auto taggedPointer = TaggedPointer::Unpack(opaquePointer);
        return taggedPointer.GetUniqueId();
    }

    void DmlReservedResourceSubAllocator::FreeResource(AllocationInfo* allocInfo, uint64_t resourceId)
    {
        // Since this allocator is warapped by ORT's BFC allocator, it's possible that the context is already
        // close at this point if the application is winding down.
        if (!m_context->Closed())
        {
            assert(allocInfo != nullptr); // Can't free nullptr

            if (allocInfo->GetOwner() != this)
            {
                // This allocation doesn't belong to this allocator!
                ORT_THROW_HR(E_INVALIDARG);
            }

            m_context->QueueReference(allocInfo);
        }
    }

    absl::optional<uint32_t> DmlReservedResourceSubAllocator::TryReserveAllocationID()
    {
        // The mutex must already be held
        assert(!m_mutex.try_lock());

        if (!m_freeAllocationIds.empty())
        {
            // Return a free ID from the pool
            uint32_t id = m_freeAllocationIds.back();
            m_freeAllocationIds.pop_back();
            return id;
        }

        static constexpr uint32_t maxAllocationID = (1 << TaggedPointer::AllocationIDBits) - 1;
        if (m_currentAllocationId == maxAllocationID)
        {
            // We've reached the maximum number of allocations!
            return absl::nullopt;
        }

        ++m_currentAllocationId;
        return m_currentAllocationId;
    }

    void DmlReservedResourceSubAllocator::ReleaseAllocationID(uint32_t id)
    {
        // The mutex must already be held
        assert(!m_mutex.try_lock());

        // Add it to the pool of free IDs
        m_freeAllocationIds.push_back(id);
    }

    D3D12BufferRegion DmlReservedResourceSubAllocator::CreateBufferRegion(
        void* opaquePointer,
        uint64_t sizeInBytes)
    {
        auto taggedPointer = TaggedPointer::Unpack(opaquePointer);

        // We need to access (mutable) state after this point, so we need to lock
        std::unique_lock<std::mutex> lock(m_mutex);

        // Find the allocation corresponding to this pointer
        auto it = m_allocationsById.find(taggedPointer.allocationId);
        ORT_THROW_HR_IF(E_INVALIDARG, it == m_allocationsById.end());

        // Make sure that we are aligned to 4 bytes to satisfy DML's requirements
        constexpr uint64_t DML_ALIGNMENT = 4;
        sizeInBytes = (1 + (sizeInBytes - 1) / DML_ALIGNMENT) * DML_ALIGNMENT;

        // Make sure the region we're trying to create fits entirely in the resource
        assert(it->second->GetD3D12Resource()->GetDesc().Width >= taggedPointer.offset + sizeInBytes);

        return D3D12BufferRegion(
            taggedPointer.offset,
            sizeInBytes,
            it->second->GetD3D12Resource());
    }

    AllocationInfo* DmlReservedResourceSubAllocator::GetAllocationInfo(void* opaquePointer)
    {
        auto taggedPointer = TaggedPointer::Unpack(opaquePointer);

        // We need to access (mutable) state after this point, so we need to lock
        std::unique_lock<std::mutex> lock(m_mutex);

        // Find the allocation corresponding to this pointer
        auto it = m_allocationsById.find(taggedPointer.allocationId);
        return it->second.Get();
    }

} // namespace Dml
