// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "ExecutionContext.h"
#include "CommandQueue.h"

namespace Dml
{
    ExecutionContext::ExecutionContext(
        ID3D12Device* d3d12Device,
        IDMLDevice* dmlDevice,
        ID3D12CommandQueue* queue
        )
        : m_queue(std::make_shared<CommandQueue>(queue))
        , m_dmlRecorder(d3d12Device, dmlDevice, m_queue)
    {
        ORT_THROW_IF_FAILED(dmlDevice->GetParentDevice(IID_GRAPHICS_PPV_ARGS(m_d3dDevice.GetAddressOf())));
    }

    void ExecutionContext::SetAllocator(std::weak_ptr<BucketizedBufferAllocator> allocator)
    {
        m_dmlRecorder.SetAllocator(allocator);
    }

    void ExecutionContext::CopyBufferRegion(
        ID3D12Resource* dstBuffer,
        uint64_t dstOffset,
        D3D12_RESOURCE_STATES dstState,
        ID3D12Resource* srcBuffer,
        uint64_t srcOffset,
        D3D12_RESOURCE_STATES srcState,
        uint64_t byteCount)
    {
        assert(!m_closed);

        SetCommandRecorder(&m_dmlRecorder);

        std::vector<D3D12_RESOURCE_BARRIER> barriers;

        if (!(dstState & D3D12_RESOURCE_STATE_COPY_DEST))
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(dstBuffer, dstState, D3D12_RESOURCE_STATE_COPY_DEST));
        }
        if (!(srcState & D3D12_RESOURCE_STATE_COPY_SOURCE))
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(srcBuffer, srcState, D3D12_RESOURCE_STATE_COPY_SOURCE));
        }

        if (!barriers.empty())
        {
            m_dmlRecorder.ResourceBarrier(barriers);
        }

        m_dmlRecorder.CopyBufferRegion(dstBuffer, dstOffset, srcBuffer, srcOffset, byteCount);

        // Reset barrier state
        if (!barriers.empty())
        {
            for (auto& barrier : barriers)
            {
                std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
            }

            m_dmlRecorder.ResourceBarrier(barriers);
        }
    }

    void ExecutionContext::FillBufferWithPattern(
        ID3D12Resource* dstBuffer,
        gsl::span<const std::byte> value /* Data type agnostic value, treated as raw bits */)
    {
        SetCommandRecorder(&m_dmlRecorder);
        m_dmlRecorder.FillBufferWithPattern(dstBuffer, value);
    }

    void ExecutionContext::ExecuteCommandList(
        ID3D12GraphicsCommandList* commandList,
        _Outptr_ ID3D12Fence** fence,
        _Out_ uint64_t* completionValue
        )
    {
        assert(!m_closed);

        SetCommandRecorder(&m_dmlRecorder);
        m_dmlRecorder.ExecuteCommandList(commandList, fence, completionValue);
    }

    void ExecutionContext::InitializeOperator(
        IDMLCompiledOperator* op,
        const DML_BINDING_DESC& persistentResourceBinding,
        const DML_BINDING_DESC& inputArrayBinding)
    {
        assert(!m_closed);
        SetCommandRecorder(&m_dmlRecorder);

        m_dmlRecorder.InitializeOperator(op, persistentResourceBinding, inputArrayBinding);
    }

    void ExecutionContext::ExecuteOperator(
        IDMLCompiledOperator* op,
        const DML_BINDING_DESC& persistentResourceBinding,
        gsl::span<const DML_BINDING_DESC> inputBindings,
        gsl::span<const DML_BINDING_DESC> outputBindings)
    {
        assert(!m_closed);
        SetCommandRecorder(&m_dmlRecorder);

        m_dmlRecorder.ExecuteOperator(op, persistentResourceBinding, inputBindings, outputBindings);
    }

    void ExecutionContext::AddUAVBarrier()
    {
        assert(!m_closed);
        SetCommandRecorder(&m_dmlRecorder);

        m_dmlRecorder.AddUAVBarrier();
    }

    void ExecutionContext::ResourceBarrier(gsl::span<const D3D12_RESOURCE_BARRIER> barriers)
    {
        assert(!m_closed);
        SetCommandRecorder(&m_dmlRecorder);

        m_dmlRecorder.ResourceBarrier(barriers);
    }

    void ExecutionContext::GetCommandListForRecordingAndInvalidateState(ID3D12GraphicsCommandList** commandList)
    {
        assert(!m_closed);
        SetCommandRecorder(&m_dmlRecorder);

        // Ensure the descriptor heap is reset to D3D as something external may change it before recording
        m_dmlRecorder.InvalidateDescriptorHeap();

        m_dmlRecorder.GetCommandList().CopyTo(commandList);
    }

    void ExecutionContext::SetCommandRecorder(ICommandRecorder* newRecorder)
    {
        assert(!m_closed);

        // If changing which recorder is the current one, we need to flush the old one first. This is to ensure correct
        // ordering of operations on the command queue.
        if (m_currentRecorder != newRecorder)
        {
            Flush();
            m_currentRecorder = newRecorder;

            if (m_currentRecorder != nullptr)
            {
                m_currentRecorder->Open();
            }
        }
    }

    void ExecutionContext::Flush()
    {
        assert(!m_closed);

        if (!m_currentRecorder || !m_currentRecorder->HasUnsubmittedWork())
        {
            // Nothing to flush
            return;
        }

        m_currentRecorder->CloseAndExecute();
        ReleaseCompletedReferences();

        // Pre-emptively set the DML command recorder.  It's the only command recorder right now,
        // and doing this here causes work and allocations resetting the command list to occur at
        // a point where it's going to be parallelized with GPU work.
        m_currentRecorder = nullptr;
        SetCommandRecorder(&m_dmlRecorder);
    }

    void ExecutionContext::QueueReference(IUnknown* object)
    {
        assert(!m_closed);
        // If something has been recorded into a command list but not submitted yet, it means that the *next* fence
        // value is the one to signal completion.
        bool waitForUnsubmittedWork = (m_currentRecorder != nullptr);
        m_queue->QueueReference(object, waitForUnsubmittedWork);
    }

    void ExecutionContext::Close()
    {
        assert(!m_closed);

        // Discard unflushed work and clear queued references.  This prevents the circular reference:
        // Kernel --> ProviderImpl -->  Context --> QueuedRefs --> Kernel
        m_queue->Close();
        m_currentRecorder = nullptr;
        m_closed = true;
    }

    GpuEvent ExecutionContext::GetCurrentCompletionEvent()
    {
        assert(!m_closed);

        GpuEvent event = m_queue->GetCurrentCompletionEvent();

        // If something has been recorded into a command list but not submitted yet, it means that the *next* fence
        // value is the one to signal completion.
        const bool unflushedWorkExists = (m_currentRecorder != nullptr) && m_currentRecorder->HasUnsubmittedWork();
        if (unflushedWorkExists)
        {
            ++event.fenceValue;
        }

        return event;
    }

    void ExecutionContext::ReleaseCompletedReferences()
    {
        assert(!m_closed);
        m_queue->ReleaseCompletedReferences();
    }

    D3D12_COMMAND_LIST_TYPE ExecutionContext::GetCommandListTypeForQueue() const
    {
        assert(!m_closed);
        return m_queue->GetType();
    }

} // namespace Dml
