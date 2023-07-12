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
        Microsoft::WRL::ComPtr<ID3D12Resource> resource_uav_state;
    };
}
