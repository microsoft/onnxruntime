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

        // The smallest bucket is 2^n bytes large, where n = c_minResourceSizeExponent
        index = std::max<gsl::index>(index, c_minResourceSizeExponent);
        index -= c_minResourceSizeExponent;

        return index;
    }

    /*static*/ uint64_t DmlReservedResourceSubAllocator::GetBucketSizeFromIndex(gsl::index index)
    {
        return (1ull << (index + c_minResourceSizeExponent));
    }

    void DmlReservedResourceSubAllocator::SetDefaultRoundingMode(AllocatorRoundingMode roundingMode)
    {
        m_defaultRoundingMode = roundingMode;
    }

    static bool GetTilingEnabled(ID3D12Device* device)
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
        if (SUCCEEDED(device->CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                &options,
                sizeof(options))))
        {
            return options.TiledResourcesTier >= D3D12_TILED_RESOURCES_TIER_1;
        }

        return false;
    }

    static uint64_t GetMaxHeapSizeInTiles()
    {
        return DmlReservedResourceSubAllocator::kDefaultMaxHeapSizeInTiles;
    }

    DmlReservedResourceSubAllocator::DmlReservedResourceSubAllocator(
        ID3D12Device* device,
        std::shared_ptr<ExecutionContext> context,
        ID3D12CommandQueue* queue,
        const D3D12_HEAP_PROPERTIES& heap_props,
        D3D12_HEAP_FLAGS heap_flags,
        D3D12_RESOURCE_FLAGS resource_flags,
        D3D12_RESOURCE_STATES initial_state)
        : m_device(device),
        m_context(context),
        queue_(queue),
        heap_properties_(heap_props),
        heap_flags_(heap_flags),
        resource_flags_(resource_flags),
        initial_state_(initial_state),
        tiling_enabled_(GetTilingEnabled(device)),
        max_heap_size_in_tiles_(GetMaxHeapSizeInTiles())
    {
    }

    absl::optional<DmlHeapAllocation> DmlReservedResourceSubAllocator::TryCreateTiledAllocation(uint64_t size_in_bytes)
    {
        DmlHeapAllocation allocation = {};

        // The allocation may be larger than the requested size to ensure a whole
        // number of tiles.
        const uint64_t resource_size_in_tiles = 1 + (size_in_bytes - 1) / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
        const uint64_t resource_size_in_bytes = resource_size_in_tiles * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
        auto resource_desc = CD3DX12_RESOURCE_DESC::Buffer(resource_size_in_bytes, resource_flags_);

        HRESULT create_resource_hr = m_device->CreateReservedResource(
            &resource_desc,
            initial_state_,
            nullptr,
            IID_PPV_ARGS(&allocation.resource_uav_state));

        if (create_resource_hr == E_OUTOFMEMORY)
        {
            return absl::nullopt;
        }
        ORT_THROW_IF_FAILED(create_resource_hr);

        // Reserve enough heaps to store all tiles in the resource.
        const uint64_t heap_count = 1 + (resource_size_in_tiles - 1) / max_heap_size_in_tiles_;
        allocation.heaps.resize(heap_count);

        // Create heaps and map them to the primary reserved resource.
        D3D12_TILED_RESOURCE_COORDINATE resource_region_start_coordinates = {};
        uint64_t unmapped_resource_tiles = resource_size_in_tiles;
        for (uint64_t i = 0; i < heap_count; i++)
        {
            // Create heap. The last heap of the allocation may have fewer tiles to
            // avoid wasting space.
            uint64_t heap_size_in_tiles = std::min<uint64_t>(
                unmapped_resource_tiles,
                max_heap_size_in_tiles_);
            uint64_t heap_size_in_bytes =
                heap_size_in_tiles * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
            auto heap_desc = CD3DX12_HEAP_DESC(
                heap_size_in_bytes,
                heap_properties_,
                0,
                heap_flags_);

            HRESULT create_heap_hr =
                m_device->CreateHeap(&heap_desc, IID_PPV_ARGS(&allocation.heaps[i]));
            if (create_heap_hr == E_OUTOFMEMORY)
            {
                return absl::nullopt;
            }
            ORT_THROW_IF_FAILED(create_heap_hr);

            // Source region in the resource to map.
            D3D12_TILE_REGION_SIZE resource_region_size = {};
            resource_region_size.NumTiles = static_cast<uint32_t>(heap_size_in_tiles);

            // Target range in the current heap to map.
            const D3D12_TILE_RANGE_FLAGS tile_range_flags =
                D3D12_TILE_RANGE_FLAG_NONE;
            const uint32_t heap_range_start_offset = 0;
            const uint32_t heap_range_tile_count = static_cast<uint32_t>(heap_size_in_tiles);

            constexpr uint32_t numResourceRegions = 1;
            constexpr uint32_t numHeapRanges = 1;

            // This is a brand new allocation/resource, so the tile mappings are
            // guaranteed to be set (on the GPU timeline) by the time any code can
            // reference the returned resource. We only execute operations on a
            // single hardware queue so there is no need to wait or signal.
            queue_->UpdateTileMappings(
                allocation.resource_uav_state.Get(),
                numResourceRegions,
                &resource_region_start_coordinates,
                &resource_region_size,
                allocation.heaps[i].Get(),
                numHeapRanges,
                &tile_range_flags,
                &heap_range_start_offset,
                &heap_range_tile_count,
                D3D12_TILE_MAPPING_FLAG_NONE);

            resource_region_start_coordinates.X += static_cast<uint32_t>(heap_size_in_tiles);
            unmapped_resource_tiles -= heap_size_in_tiles;
        }

        assert(unmapped_resource_tiles == 0);

        return allocation;
    }

    absl::optional<DmlHeapAllocation> DmlReservedResourceSubAllocator::TryCreateUntiledAllocation(uint64_t size_in_bytes)
    {
        DmlHeapAllocation allocation = {};

        // Create the allocation's sole heap. The allocation may be larger than the
        // requested size to ensure a whole number of tiles.
        allocation.heaps.resize(1);
        D3D12_HEAP_DESC heap_desc =
            CD3DX12_HEAP_DESC(size_in_bytes, heap_properties_, 0, heap_flags_);
        HRESULT create_heap_hr = m_device->CreateHeap(
            &heap_desc,
            IID_PPV_ARGS(&allocation.heaps.front()));
        if (create_heap_hr == E_OUTOFMEMORY)
        {
            return absl::nullopt;
        }

        // Create large placed resource that spans the heap.
        D3D12_RESOURCE_DESC resource_desc = CD3DX12_RESOURCE_DESC::Buffer(size_in_bytes, resource_flags_);

        HRESULT create_resource_hr = m_device->CreatePlacedResource(
            allocation.heaps.front().Get(),
            0,
            &resource_desc,
            initial_state_,
            nullptr,
            IID_PPV_ARGS(&allocation.resource_uav_state));
        if (create_resource_hr == E_OUTOFMEMORY)
        {
            return absl::nullopt;
        }
        ORT_THROW_IF_FAILED(create_resource_hr);

        return allocation;
    }

    uint64_t DmlReservedResourceSubAllocator::ComputeRequiredSize(size_t size)
    {
        const uint64_t resource_size_in_tiles =
            1 + (size - 1) / D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
        const uint64_t resource_size_in_bytes =
            resource_size_in_tiles * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;

        return resource_size_in_bytes;
    }

    void* DmlReservedResourceSubAllocator::Alloc(size_t size_in_bytes)
    {
        // For some reason lotus likes requesting 0 bytes of memory
        size_in_bytes = std::max<size_t>(1, size_in_bytes);

        // The D3D12 device is thread-safe so we don't need to hold the lock while
        // creating an allocation.
        absl::optional<DmlHeapAllocation> allocation =
            tiling_enabled_ ? TryCreateTiledAllocation(size_in_bytes)
                            : TryCreateUntiledAllocation(size_in_bytes);

        ORT_THROW_HR_IF(E_INVALIDARG, !allocation);

        // We need to access (mutable) state after this point, so we need to lock
        std::unique_lock<std::mutex> lock(mutex_);

        absl::optional<uint32_t> allocationId = TryReserveAllocationID();
        ORT_THROW_HR_IF(E_INVALIDARG, !allocationId);

        auto resourceWrapper = wil::MakeOrThrow<DmlReservedResourceWrapper>(std::move(*allocation));
        ComPtr<AllocationInfo> allocInfo = wil::MakeOrThrow<AllocationInfo>(
            this,
            ++m_currentAllocationId,
            resourceWrapper.Get(),
            size_in_bytes
        );

        allocations_by_id_.emplace(*allocationId, allocInfo);

        lock.unlock();

    #if _DEBUG
        m_outstandingAllocationsById[allocInfo->GetId()] = allocInfo.Get();
    #endif

        // DML only has a single device in ORT at the moment
        const uint64_t device_id = 0;
        const uint64_t offset = 0;
        return TaggedPointer::Pack(device_id, *allocationId, offset);
    }

    void DmlReservedResourceSubAllocator::Free(void* ptr)
    {
        ORT_THROW_HR_IF(E_INVALIDARG, ptr == nullptr);

        TaggedPointer tagged_ptr = TaggedPointer::Unpack(ptr);
        ORT_THROW_HR_IF(E_INVALIDARG, tagged_ptr.offset != 0);

        // We need to access (mutable) state after this point, so we need to lock
        std::unique_lock<std::mutex> lock(mutex_);

        auto it = allocations_by_id_.find(tagged_ptr.allocation_id);
        ORT_THROW_HR_IF(E_INVALIDARG, it == allocations_by_id_.end());

        ReleaseAllocationID(tagged_ptr.allocation_id);

        // Frees the ID3D12Heap
        allocations_by_id_.erase(it);
    }

    void DmlReservedResourceSubAllocator::FreeResource(AllocationInfo* allocInfo)
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
        assert(!mutex_.try_lock());

        if (!free_allocation_ids_.empty())
        {
            // Return a free ID from the pool
            uint32_t id = free_allocation_ids_.back();
            free_allocation_ids_.pop_back();
            return id;
        }

        static constexpr uint32_t kMaxAllocationID =
            (1 << TaggedPointer::kAllocationIDBits) - 1;
        if (current_allocation_id_ == kMaxAllocationID)
        {
            // We've reached the maximum number of allocations!
            return absl::nullopt;
        }

        ++current_allocation_id_;
        return current_allocation_id_;
    }

    void DmlReservedResourceSubAllocator::ReleaseAllocationID(uint32_t id)
    {
        // The mutex must already be held
        assert(!mutex_.try_lock());

        // Add it to the pool of free IDs
        free_allocation_ids_.push_back(id);
    }

    D3D12BufferRegion DmlReservedResourceSubAllocator::CreateBufferRegion(
        const TaggedPointer& taggedPointer,
        uint64_t size_in_bytes)
    {
        // We need to access (mutable) state after this point, so we need to lock
        std::unique_lock<std::mutex> lock(mutex_);

        // Find the allocation corresponding to this pointer
        auto it = allocations_by_id_.find(taggedPointer.allocation_id);
        ORT_THROW_HR_IF(E_INVALIDARG, it == allocations_by_id_.end());

        // Make sure that we are aligned to 4 bytes to satisfy DML's requirements
        constexpr uint64_t DML_ALIGNMENT = 4;
        size_in_bytes =
            (1 + (size_in_bytes - 1) / DML_ALIGNMENT) * DML_ALIGNMENT;

        // Make sure the region we're trying to create fits entirely in the resource
        assert(it->second->GetUavResource()->GetDesc().Width >= taggedPointer.offset + size_in_bytes);

        return D3D12BufferRegion(
            taggedPointer.offset,
            size_in_bytes,
            it->second->GetUavResource());
    }

    AllocationInfo* DmlReservedResourceSubAllocator::GetAllocationInfo(const TaggedPointer& taggedPointer)
    {
        // We need to access (mutable) state after this point, so we need to lock
        std::unique_lock<std::mutex> lock(mutex_);

        // Find the allocation corresponding to this pointer
        auto it = allocations_by_id_.find(taggedPointer.allocation_id);
        return it->second.Get();
    }

} // namespace Dml
