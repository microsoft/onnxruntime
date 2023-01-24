// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Dml
{
    struct DmlHeapAllocation
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
}
