// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ICommandRecorder.h"
#include "CommandAllocatorRing.h"

namespace Dml
{
    class CommandQueue;
    class BucketizedBufferAllocator;

    class DmlCommandRecorder : public ICommandRecorder
    {
    public:
        DmlCommandRecorder(
            ID3D12Device* d3dDevice,
            IDMLDevice* device, 
            std::shared_ptr<CommandQueue> commandQueue);

        void InitializeOperator(
            IDMLCompiledOperator* op,
            const DML_BINDING_DESC& persistentResourceBinding,
            const DML_BINDING_DESC& inputArrayBinding);

        void ExecuteOperator(
            IDMLCompiledOperator* op,
            const DML_BINDING_DESC& persistentResourceBinding,
            gsl::span<const DML_BINDING_DESC> inputBindings,
            gsl::span<const DML_BINDING_DESC> outputBindings);

        void CopyBufferRegion(
            ID3D12Resource* dstBuffer,
            uint64_t dstOffset,
            ID3D12Resource* srcBuffer,
            uint64_t srcOffset,
            uint64_t byteCount);

        void FillBufferWithPattern(
            ID3D12Resource* dstBuffer,
            gsl::span<const std::byte> value /* Data type agnostic value, treated as raw bits */);

        void ExecuteCommandList(
            ID3D12GraphicsCommandList* commandList,
            _Outptr_ ID3D12Fence** fence,
            _Out_ uint64_t* completionValue);

        ComPtr<ID3D12GraphicsCommandList> GetCommandList();
        
        void ResourceBarrier(gsl::span<const D3D12_RESOURCE_BARRIER> barriers);
        void AddUAVBarrier();

        void Open() final;
        void CloseAndExecute() final;
        
        void SetAllocator(std::weak_ptr<BucketizedBufferAllocator> allocator);

        bool HasUnsubmittedWork() override
        {
            return m_operationsRecordedInCurrentCommandList || !m_pendingCommandLists.empty();
        }

    private:

        std::shared_ptr<CommandQueue> m_queue;
        ComPtr<ID3D12Device> m_d3dDevice;
        ComPtr<IDMLDevice> m_dmlDevice;
        ComPtr<IDMLOperatorInitializer> m_initializer;
        ComPtr<IDMLCommandRecorder> m_recorder;

        // Descriptors are allocated from a pool. The current heap pointer is only used to avoid redundantly
        // setting the same heap; it does not have ownership of the heap object.
        DescriptorPool m_descriptorPool;
        ID3D12DescriptorHeap* m_currentDescriptorHeap = nullptr;

        // The weak pointer avoids a circular reference from context->recorder->allocator->context
        std::weak_ptr<BucketizedBufferAllocator> m_bufferAllocator;

        CommandAllocatorRing<2> m_commandAllocatorRing;

        // The command list currently being recorded into, and whether any command have been recorded yet.
        ComPtr<ID3D12GraphicsCommandList> m_currentCommandList;
        bool m_operationsRecordedInCurrentCommandList = false;

        // Command lists which have been batched up for execution.  The values in 
        // m_pendingCommandListsCacheable indicate whether they can be moved into this
        // class's cache after execution, versus if they belong to the caller and were
        // passed to ExecuteCommandList.
        std::vector<ComPtr<ID3D12GraphicsCommandList>> m_pendingCommandLists;
        std::vector<bool> m_pendingCommandListsCacheable;

        // A pool of cached command lists which may be re-used.
        std::deque<ComPtr<ID3D12GraphicsCommandList>> m_cachedCommandLists;

        void SetDescriptorHeap(ID3D12DescriptorHeap* descriptorHeap);
    };

} // namespace Dml
