// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "GpuEvent.h"

namespace Dml
{
    // A fixed-size ring of command allocators. Each time an allocator is retrieved, the allocator will
    // be reset if its previously recorded commands have finished executing on the GPU.
    template <size_t AllocatorCount>
    class CommandAllocatorRing
    {
    public:
        CommandAllocatorRing(
            ID3D12Device* device, 
            D3D12_COMMAND_LIST_TYPE commandListType, 
            GpuEvent initialEvent)
        {
            for (auto& info : m_commandAllocators)
            {
                ORT_THROW_IF_FAILED(device->CreateCommandAllocator(
                    commandListType,
                    IID_PPV_ARGS(&info.allocator)));

                info.completionEvent = initialEvent;
            }
        }

        ID3D12CommandAllocator* GetCurrentAllocator()
        {
            CommandAllocatorInfo& allocatorInfo = m_commandAllocators[m_currentCommandAllocator];

            // Take the opportunity to reset the command allocator if possible.
            if (allocatorInfo.completionEvent.IsSignaled())
            {
                ORT_THROW_IF_FAILED(allocatorInfo.Get()->Reset());
            }

            return m_commandAllocators[m_currentCommandAllocator].Get();
        }

        void AdvanceAllocator(GpuEvent completionEvent)
        {
            // Set the completion event for the current allocator so it can be reset eventually.
            m_commandAllocators[m_currentCommandAllocator].completionEvent = completionEvent;

            // Advance to the next allocator.
            m_currentCommandAllocator = (m_currentCommandAllocator + 1) % AllocatorCount;
        }

    private:
        struct CommandAllocatorInfo
        {
            ComPtr<ID3D12CommandAllocator> allocator;

            // The event which will be signaled when the last command list submitted using this allocator
            // completes execution on the GPU.
            GpuEvent completionEvent = {};

            ID3D12CommandAllocator* Get() const { return allocator.Get(); }
        };

        std::array<CommandAllocatorInfo, AllocatorCount> m_commandAllocators;
        size_t m_currentCommandAllocator = 0;

    };
}