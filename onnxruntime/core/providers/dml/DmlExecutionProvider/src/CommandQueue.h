// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "GpuEvent.h"

namespace Dml
{
    // Manages a D3D12 command queue and provides a waitable fence which is signaled with a monotonically increasing
    // value once each execute completes on the GPU.
    class CommandQueue
    {
    public:
        // Creates a CommandQueue object that wraps an existing D3D12 queue.
        CommandQueue(ID3D12CommandQueue* existingQueue);

        D3D12_COMMAND_LIST_TYPE GetType() const { return m_type; }
        ComPtr<ID3D12Fence> GetFence() const { return m_fence; }
        uint64_t GetLastFenceValue() const { return m_lastFenceValue; }

        void ExecuteCommandList(ID3D12CommandList* commandList);
        void ExecuteCommandLists(gsl::span<ID3D12CommandList*> commandLists);

        // Queues a wait to block the GPU until the specified fence is signaled to a given value.
        void Wait(ID3D12Fence* fence, uint64_t value);

        // Returns an event that will become signaled when everything submitted to the queue thus far has
        // completed execution on the GPU.
        GpuEvent GetCurrentCompletionEvent();

        // Returns an event that will become signaled after the next ExecuteCommandLists call.
        GpuEvent GetNextCompletionEvent();

        void QueueReference(IUnknown* object, bool waitForUnsubmittedWork);

#ifdef _GAMING_XBOX
        void QueueReference(IGraphicsUnknown* object, bool waitForUnsubmittedWork)
        {
            // TODO(justoeck): consider changing QueuedReference to hold a variant of
            // ComPtr<IUnknown>, ComPtr<IGraphicsUnknown>.
            auto wrapper = Microsoft::WRL::Make<GraphicsUnknownWrapper>(object);
            QueueReference(wrapper.Get(), waitForUnsubmittedWork);
        }
#endif

        void Close();
        void ReleaseCompletedReferences();

        HRESULT GetCommandQueue(ID3D12CommandQueue** queue);

    private:
        struct QueuedReference
        {
            uint64_t fenceValue;
            ComPtr<IUnknown> object;
        };

        std::deque<QueuedReference> m_queuedReferences;

        ComPtr<ID3D12CommandQueue> m_queue;
        D3D12_COMMAND_LIST_TYPE m_type;

        ComPtr<ID3D12Fence> m_fence;
        uint64_t m_lastFenceValue = 0;
        bool m_closing = false;
    };

} // namespace Dml
