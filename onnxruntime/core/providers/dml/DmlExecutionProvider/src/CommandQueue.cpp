// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "CommandQueue.h"

namespace Dml
{
    CommandQueue::CommandQueue(ID3D12CommandQueue* existingQueue, bool cpuSyncSpinningEnabled)
        : m_queue(existingQueue)
        , m_type(existingQueue->GetDesc().Type)
        , m_cpuSyncSpinningEnabled(cpuSyncSpinningEnabled)
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
        // If the CommandQueue is releasing completed references, we don't need to queue the reference up again.
        if (!m_closing && !m_releasing)
        {
            QueuedReference queuedReference = {GetLastFenceValue(), object};

            // If something has been recorded into a command list but not submitted yet, it means that the *next* fence
            // value is the one to signal completion.
            if (waitForUnsubmittedWork)
            {
                ++queuedReference.fenceValue;
            }

            // We don't need to queue references whose work has already completed on the GPU by the time the CPU queues
            // the reference. Just let it go out of scope.
            uint64_t completedValue = GetFence()->GetCompletedValue();
            if (queuedReference.fenceValue > completedValue)
            {
                m_queuedReferences.push_back(queuedReference);
            }
        }
    }

    void CommandQueue::Close()
    {
        // Wait for flushed work:
        assert(!m_closing);
        m_closing = true;
        GpuEvent event = GetCurrentCompletionEvent();
        event.WaitForSignal(m_cpuSyncSpinningEnabled);
        m_queuedReferences.clear();
        m_closing = false;
    }

    void CommandQueue::ReleaseCompletedReferences()
    {
        uint64_t completedValue = GetFence()->GetCompletedValue();
        m_releasing = true;
        while (!m_queuedReferences.empty() && m_queuedReferences.front().fenceValue <= completedValue)
        {
            m_queuedReferences.pop_front();
        }
        m_releasing = false;
    }

} // namespace Dml
