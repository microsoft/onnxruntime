// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <DirectML.h>

namespace Dml
{
    struct DmlReusedCommandListState
    {
        // Re-usable command list, supporting descriptor heap, and DML binding table to update that heap.
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> graphicsCommandList;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> commandAllocator;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
        Microsoft::WRL::ComPtr<IDMLBindingTable> bindingTable;
        Microsoft::WRL::ComPtr<ID3D12Resource> persistentResource;
        Microsoft::WRL::ComPtr<ID3D12Resource> temporaryResource;
        Microsoft::WRL::ComPtr<IUnknown> persistentResourceAllocatorUnknown;

        // Bindings from previous executions of a re-used command list
        mutable std::vector<uint64_t> inputBindingAllocIds;
        mutable std::vector<uint64_t> outputBindingAllocIds;
        mutable uint64_t tempBindingAllocId = 0;

        // Fence tracking the status of the command list's last execution, and whether its descriptor heap
        // can safely be updated.
        mutable Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        mutable uint64_t completionValue = 0;
    };
}
