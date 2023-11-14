// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "CommandQueue.h"

namespace Dml
{
    CommandQueue::CommandQueue(ID3D12CommandQueue* existingQueue)
        : m_queue(existingQueue)
        , m_type(existingQueue->GetDesc().Type)
    {
        ComPtr<ID3D12Device> device;
        GRAPHICS_THROW_IF_FAILED(m_queue->GetDevice(IID_GRAPHICS_PPV_ARGS(device.GetAddressOf())));
        ORT_THROW_IF_FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_GRAPHICS_PPV_ARGS(m_fence.ReleaseAndGetAddressOf())));
    }

    void CommandQueue::ExecuteCommandList(ID3D12CommandList* commandList)
    {
        ExecuteCommandLists(gsl::make_span(&commandList, 1));
    }

    void CommandQueue::ExecuteCommandLists(gsl::span<ID3D12CommandList*> commandLists)
    {
        m_queue->ExecuteCommandLists(gsl::narrow<uint32_t>(commandLists.size()), commandLists.data());

        ++m_lastFenceValue;
        ORT_THROW_IF_FAILED(m_queue->Signal(m_fence.Get(), m_lastFenceValue));
    }

    void CommandQueue::Wait(ID3D12Fence* fence, uint64_t value)
    {
        ORT_THROW_IF_FAILED(m_queue->Wait(fence, value));

        ++m_lastFenceValue;
        ORT_THROW_IF_FAILED(m_queue->Signal(m_fence.Get(), m_lastFenceValue));
    }

    GpuEvent CommandQueue::GetCurrentCompletionEvent()
    {
        return GpuEvent{ m_lastFenceValue, m_fence };
    }

    GpuEvent CommandQueue::GetNextCompletionEvent()
    {
        return GpuEvent{ m_lastFenceValue + 1, m_fence };
    }

    void CommandQueue::QueueReference(IUnknown* object, bool waitForUnsubmittedWork) 
    {
        // If the CommandQueue is closing, then m_queuedReferences is being cleared -- it is not OK 
        // to queue additional references at this time, since those references would be leaked. This 
        // affects any objects in m_queuedReferences whose destructors indirectly call QueueReference; 
        // for example, an allocation from BucketizedBufferAllocator attempts to queue a reference 
        // to its underlying D3D resource when freed. Furthermore, these references are unnecessary
        // since Close() already blocks for scheduled GPU work before clearing m_queuedReferences.
        if (!m_closing)
        {
            QueuedReference queuedReference = {GetLastFenceValue(), object};

            // If something has been recorded into a command list but not submitted yet, it means that the *next* fence
            // value is the one to signal completion.
            if (waitForUnsubmittedWork)
            {
                ++queuedReference.fenceValue;
            }

            m_queuedReferences.push_back(queuedReference);
        }
    }
    
    void CommandQueue::Close()
    {
        // Wait for flushed work:
        assert(!m_closing);
        m_closing = true;
        GpuEvent event = GetCurrentCompletionEvent();
        event.WaitForSignal();
        m_queuedReferences.clear();
        m_closing = false;
    }
    
    void CommandQueue::ReleaseCompletedReferences()
    {
        uint64_t completedValue = GetFence()->GetCompletedValue();
        while (!m_queuedReferences.empty() && m_queuedReferences.front().fenceValue <= completedValue)
        {
            m_queuedReferences.pop_front();
        }
    }

    HRESULT CommandQueue::GetCommandQueue(ID3D12CommandQueue** queue)
    {
        return m_queue.CopyTo(queue);
    }

} // namespace Dml
