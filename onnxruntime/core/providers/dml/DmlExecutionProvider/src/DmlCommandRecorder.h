// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ICommandRecorder.h"

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

        ~DmlCommandRecorder();

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
            return m_operationsRecordedInCurrentCommandList;
        }

        // Forces the descriptor heap to be reset to D3D before executing future operations
        void InvalidateDescriptorHeap()
        {
            m_currentDescriptorHeap = nullptr;
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

        void CloseAndExecute(_In_opt_ ID3D12GraphicsCommandList* commandList);

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

        // The command list currently being recorded into, and whether any command have been recorded yet.
        ComPtr<ID3D12GraphicsCommandList> m_currentCommandList;
        bool m_operationsRecordedInCurrentCommandList = false;

        static constexpr int commandListCount = 3;

        // We use enough command lists and allocators to allow command lists to be reset in a different thread while
        // there is another command list ready to receive commands. When we execute and close a command list, we start
        // the resetting process on a different thread and set m_currentCommandList to the next available one.
        std::array<ComPtr<ID3D12GraphicsCommandList>, commandListCount> m_commandListRing;
        std::array<CommandAllocatorInfo, commandListCount> m_allocatorRing;

        // We should always have 1 less reset thread than command lists since we always need a clean command list, but
        // the other ones can all be in the process of getting reset
        std::array<std::optional<std::thread>, commandListCount - 1> m_resetThreads;

        void SetDescriptorHeap(ID3D12DescriptorHeap* descriptorHeap);
    };

} // namespace Dml
