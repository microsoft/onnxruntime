// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "CommandAllocatorRing.h"

namespace Dml
{
    class DmlAllocator;
    class CommandQueue;

    // Helper that manages and wraps an ID3D12GraphicsCommandList and its backing
    // command allocator. This class is NOT thread safe.
    class DmlCommandList
    {
    public:
        // Constructs a command list.
        DmlCommandList(
            ID3D12Device* d3d12_device,
            IDMLDevice* dml_device,
            std::shared_ptr<CommandQueue> queue);

        // Records a CopyBufferRegion (see
        // ID3D12GraphicsCommandList::CopyBufferRegion) for execution. Transition
        // barriers are automatically inserted to transition the source and
        // destination resources to COPY_SOURCE and COPY_DEST if necessary.
        void CopyBufferRegion(
            ID3D12Resource* dst_buffer,
            uint64_t dst_offset,
            D3D12_RESOURCE_STATES dst_state,
            ID3D12Resource* src_buffer,
            uint64_t src_offset,
            D3D12_RESOURCE_STATES src_state,
            uint64_t byte_count);

        // Records a ClearUAV with the specified value into the command list.
        void FillBufferWithPattern(
            ID3D12Resource* dst,
            uint64_t dst_offset,
            uint64_t dst_size_in_bytes,
            absl::Span<const std::byte> value /* Data type agnostic value, treated as raw bits */);

        // Records DML operator initialization into the command list. It's safe to
        // release the binding table immediately after this is called.
        void InitializeOperator(
            IDMLOperatorInitializer* initializer,
            IDMLBindingTable* binding_table,
            ID3D12DescriptorHeap* descriptor_heap);

        // Records DML operator execution into the command list. It's safe to
        // release the binding table immediately after this is called.
        void ExecuteOperator(
            IDMLCompiledOperator* op,
            IDMLBindingTable* binding_table,
            ID3D12DescriptorHeap* descriptor_heap);

        // Records a resoruce barrier into the command list.
        void ResourceBarrier(absl::Span<const D3D12_RESOURCE_BARRIER> barriers);

        // Records a UAV barrier on all resources into the command list.
        void UavBarrier();

        // Opens the command list for recording, which is required before any of the
        // above methods can be called.
        void Open();

        // Closes the command list for recording, which is required before the
        // command list can be executed on a command queue. If any errors occur
        // while recording they will be reported as a status here.
        Status Close();

        // Returns a pointer to the underlying D3D command list.
        ID3D12CommandList* Get() { return d3d_command_list_.Get(); }

        // Forces the descriptor heap to be reset to D3D before executing future operations
        void InvalidateDescriptorHeap()
        {
            current_descriptor_heap_ = nullptr;
        }

        ComPtr<ID3D12GraphicsCommandList> GetCommandList() const
        {
            return d3d_command_list_;
        }

        DescriptorPool& GetDescriptorPool() { return descriptor_pool_; }

    private:
        Microsoft::WRL::ComPtr<ID3D12Device> d3d_device_;
        Microsoft::WRL::ComPtr<IDMLDevice> dml_device_;
        Microsoft::WRL::ComPtr<IDMLCommandRecorder> recorder_;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> d3d_command_list_;
        std::shared_ptr<CommandQueue> queue_;

        // Descriptors are allocated from a pool. The current heap pointer is only
        // used to avoid redundantly setting the same heap; it does not have
        // ownership of the heap object.
        DescriptorPool descriptor_pool_;
        ID3D12DescriptorHeap* current_descriptor_heap_ = nullptr;
        GpuEvent current_completion_event_;

        CommandAllocatorRing<2> command_allocator_ring_;

        void SetDescriptorHeap(ID3D12DescriptorHeap* descriptor_heap);
    };

} // namespace Dml
