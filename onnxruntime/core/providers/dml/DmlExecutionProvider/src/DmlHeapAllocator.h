// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlResourceWrapper.h"

namespace Dml
{

struct Allocation
{
    Microsoft::WRL::ComPtr<ID3D12Heap> heap;

    // Heaps backing the memory for the allocation. If tiling is supported
    // an allocation may comprise multiple heaps. If tiling is not supported
    // an allocation will only have a single heap.
    std::vector<Microsoft::WRL::ComPtr<ID3D12Heap>> heaps;

    // Resources created over this allocation's heaps. All three resources
    // are identical aside from being fixed in a single resource state: UAV,
    // COPY_SRC, and COPY_DST respectively. The purpose of duplicate
    // resources is to enable overlapping resources in different states for
    // copying data. Most callers will not (and should not) interact
    // directly with these resources; all three are wrapped by the buffer
    // regions returned from this allocator, and the appropriate resource
    // will be used automatically when performing buffer copies.
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_uav_state;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_copy_src_state;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource_copy_dst_state;
};

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
class D3D12HeapAllocator
{
  public:
    // Maximum size of a heap (in tiles) when allocations are tiled. Each tile
    // is 64KB. A default size of 512 tiles (32MB) does a good job of handling
    // local video memory fragmentation without requiring lots of heaps.
    static constexpr uint64_t kDefaultMaxHeapSizeInTiles = 512;

    D3D12HeapAllocator(
        ID3D12Device* device,
        ID3D12CommandQueue* queue,
        const D3D12_HEAP_PROPERTIES& heap_props,
        D3D12_HEAP_FLAGS heap_flags,
        D3D12_RESOURCE_FLAGS resource_flags,
        D3D12_RESOURCE_STATES initial_state);

    Microsoft::WRL::ComPtr<DmlResourceWrapper> Alloc(size_t size_in_bytes);
    uint64_t ComputeRequiredSize(size_t size);
    bool TilingEnabled() const { return tiling_enabled_; };

  private:
    std::mutex mutex_;

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    const D3D12_HEAP_PROPERTIES heap_properties_;
    const D3D12_HEAP_FLAGS heap_flags_;
    const D3D12_RESOURCE_FLAGS resource_flags_;
    const D3D12_RESOURCE_STATES initial_state_;
    bool tiling_enabled_;
    uint64_t max_heap_size_in_tiles_;

  private:
    absl::optional<Allocation> TryCreateTiledAllocation(uint64_t size_in_bytes);
    absl::optional<Allocation> TryCreateUntiledAllocation(
        uint64_t size_in_bytes);

    friend class D3D12BufferRegion;
};

}
