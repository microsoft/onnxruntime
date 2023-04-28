// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "ExecutionContext.h"
#include "CommandQueue.h"
#include "BucketizedBufferAllocator.h"

namespace Dml
{
    ExecutionContext::ExecutionContext(
        ID3D12Device* d3d_device,
        IDMLDevice* dml_device,
        ID3D12CommandQueue* queue)
        : m_dmlDevice(dml_device)
    {
        ORT_THROW_IF_FAILED(dml_device->CreateOperatorInitializer(0, nullptr, IID_PPV_ARGS(&m_initializer)));

        dml_command_queue_ = std::make_shared<CommandQueue>(queue);

        batch_state_ = std::make_shared<BatchState>();
        batch_state_->next_flush_event = dml_command_queue_->GetCurrentCompletionEvent();
        ++batch_state_->next_flush_event.fenceValue;

        uint32_t batch_flush_size = default_batch_flush_size;
        uint32_t batch_flush_time_us = default_batch_flush_time_us;

        dml_command_list_ = std::make_shared<DmlCommandList>(
            d3d_device,
            dml_device,
            dml_command_queue_);

        execution_thread_ = std::thread(
            ExecutionThreadProc,
            batch_state_,
            dml_command_list_,
            dml_command_queue_,
            batch_flush_size,
            batch_flush_time_us);
    }

    void ExecutionContext::SetAllocator(std::weak_ptr<BucketizedBufferAllocator> allocator)
    {
        m_bufferAllocator = allocator;
    }

    ExecutionContext::~ExecutionContext()
    {
        // Request exit of the background thread
        std::unique_lock<std::mutex> lock(batch_state_->mutex);
        batch_state_->exit_requested = true;
        batch_state_->command_added.notify_all(); // wake the thread
        lock.unlock();

        // detach() rather than join(), because we don't want (or need) to wait for
        // it to complete. This prevents blocking in a destructor, which would be
        // bad.
        execution_thread_.detach();
    }

    GpuEvent ExecutionContext::CopyBufferRegionRaw(
        ID3D12Resource* dst_buffer,
        uint64_t dst_offset,
        D3D12_RESOURCE_STATES dst_state,
        ID3D12Resource* src_buffer,
        uint64_t src_offset,
        D3D12_RESOURCE_STATES src_state,
        uint64_t byte_count)
    {
        std::unique_lock<std::mutex> lock(batch_state_->mutex);

        batch_state_->WriteBatch().emplace_back(
            [=](DmlCommandList& command_list)
            {
                command_list.CopyBufferRegion(
                    dst_buffer,
                    dst_offset,
                    dst_state,
                    src_buffer,
                    src_offset,
                    src_state,
                    byte_count);
            });

        batch_state_->command_added.notify_all();

        return batch_state_->next_flush_event;
    }

    GpuEvent ExecutionContext::CopyBufferRegion(
        ID3D12Resource* dst_buffer,
        uint64_t dst_offset,
        D3D12_RESOURCE_STATES dst_state,
        ID3D12Resource* src_buffer,
        uint64_t src_offset,
        D3D12_RESOURCE_STATES src_state,
        uint64_t byte_count)
    {
        return CopyBufferRegionRaw(
            dst_buffer,
            dst_offset,
            dst_state,
            src_buffer,
            src_offset,
            src_state,
            byte_count);
    }

    GpuEvent ExecutionContext::FillBufferWithPatternRaw(
        ID3D12Resource* dst,
        uint64_t dst_offset,
        uint64_t dst_size_in_bytes,
        absl::Span<const std::byte> value)
    {
        std::unique_lock<std::mutex> lock(batch_state_->mutex);

        absl::InlinedVector<std::byte, 16> value_copy(value.begin(), value.end());
        batch_state_->WriteBatch().emplace_back(
            [=, value = std::move(value_copy)](DmlCommandList& command_list)
            {
                command_list.FillBufferWithPattern(
                    dst,
                    dst_offset,
                    dst_size_in_bytes,
                    value);
            });

        batch_state_->command_added.notify_all();

        return batch_state_->next_flush_event;
    }

    GpuEvent ExecutionContext::FillBufferWithPattern(
        ID3D12Resource* dstBuffer,
        gsl::span<const std::byte> value /* Data type agnostic value, treated as raw bits */)
    {
        return FillBufferWithPatternRaw(
            dstBuffer,
            0,
            dstBuffer->GetDesc().Width,
            value);
    }

    void ExecutionContext::ExecuteCommandList(
        ID3D12GraphicsCommandList* commandList,
        _Outptr_ ID3D12Fence** fence,
        _Out_ uint64_t* completionValue)
    {
        std::unique_lock<std::mutex> lock(batch_state_->mutex);

        // We have to close and execute the current command list right away, regardless of whether a flush
        // was requested or not
        auto& batch = batch_state_->WriteBatch();

        if (!batch.empty())
        {
            batch_state_->flush_requested = false;
            RecordAndExecute(dml_command_queue_.get(), dml_command_list_.get(), batch);
        }

        // The caller can re-use relevant resources after the next set of work to be
        // flushed has completed.  Its command list hasn't been executed yet, just batched.
        GpuEvent gpuEvent = dml_command_queue_->GetNextCompletionEvent();
        gpuEvent.fence.CopyTo(fence);
        *completionValue = gpuEvent.fenceValue;

        dml_command_queue_->ExecuteCommandList(commandList);
    }

    GpuEvent ExecutionContext::InitializeOperator(
        IDMLCompiledOperator* op,
        const DML_BINDING_DESC& persistentResourceBinding,
        const DML_BINDING_DESC& inputArrayBinding)
    {
        // Reset the initializer to reference the input operator.
        IDMLCompiledOperator* ops[] = { op };
        ORT_THROW_IF_FAILED(m_initializer->Reset(ARRAYSIZE(ops), ops));

        DML_BINDING_PROPERTIES initBindingProps = m_initializer->GetBindingProperties();

        const uint32_t numDescriptors = initBindingProps.RequiredDescriptorCount;
        DescriptorRange descriptorRange = dml_command_list_->GetDescriptorPool().AllocDescriptors(
            numDescriptors,
            dml_command_queue_->GetNextCompletionEvent());

        // Create a binding table for initialization.
        DML_BINDING_TABLE_DESC bindingTableDesc = {};
        bindingTableDesc.Dispatchable = m_initializer.Get();
        bindingTableDesc.CPUDescriptorHandle = descriptorRange.cpuHandle;
        bindingTableDesc.GPUDescriptorHandle = descriptorRange.gpuHandle;
        bindingTableDesc.SizeInDescriptors = numDescriptors;

        ComPtr<IDMLBindingTable> bindingTable;
        ORT_THROW_IF_FAILED(m_dmlDevice->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));

        // Create a temporary resource for initializing the op, if it's required.
        UINT64 temporaryResourceSize = initBindingProps.TemporaryResourceSize;
        if (temporaryResourceSize > 0)
        {
            auto allocator = m_bufferAllocator.lock();

            // Allocate and immediately free a temporary buffer. The buffer resource will still be
            // alive (managed by the pool); freeing allows the resource to be shared with other operators.
            void* tempResourceHandle = allocator->Alloc(static_cast<size_t>(temporaryResourceSize), AllocatorRoundingMode::Enabled);
            if (!tempResourceHandle)
            {
                ORT_THROW_HR(E_OUTOFMEMORY);
            }

            ID3D12Resource* buffer = allocator->DecodeDataHandle(tempResourceHandle)->GetResource();
            allocator->Free(tempResourceHandle);

            // Bind the temporary resource.
            DML_BUFFER_BINDING bufferBinding = { buffer, 0, temporaryResourceSize };
            DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
            bindingTable->BindTemporaryResource(&bindingDesc);
        }

        // Bind inputs, if provided.
        if (inputArrayBinding.Type != DML_BINDING_TYPE_NONE)
        {
            // An operator with inputs to bind MUST use a BUFFER_ARRAY.
            assert(inputArrayBinding.Type == DML_BINDING_TYPE_BUFFER_ARRAY);
            bindingTable->BindInputs(1, &inputArrayBinding);
        }

        // Bind the persistent resource, which is an output of initialization.
        if (persistentResourceBinding.Type != DML_BINDING_TYPE_NONE)
        {
            // Persistent resources MUST be bound as buffers.
            assert(persistentResourceBinding.Type == DML_BINDING_TYPE_BUFFER);
            bindingTable->BindOutputs(1, &persistentResourceBinding);
        }

        std::unique_lock<std::mutex> lock(batch_state_->mutex);

        batch_state_->WriteBatch().emplace_back(
            [=,
            binding_table = std::move(bindingTable),
            initializer = m_initializer](DmlCommandList& command_list)
            {
                command_list.InitializeOperator(
                    initializer.Get(),
                    binding_table.Get(),
                    descriptorRange.heap);
            });

        batch_state_->command_added.notify_all();

        return batch_state_->next_flush_event;
    }

    GpuEvent ExecutionContext::ExecuteOperator(
        IDMLCompiledOperator* op,
        Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
        ID3D12DescriptorHeap* descriptor_heap)
    {
        std::unique_lock<std::mutex> lock(batch_state_->mutex);

        batch_state_->WriteBatch().emplace_back(
            [=, binding_table = std::move(binding_table)](
                DmlCommandList& command_list) {
                command_list.ExecuteOperator(
                    op,
                    binding_table.Get(),
                    descriptor_heap);
            });

        batch_state_->command_added.notify_all();

        return batch_state_->next_flush_event;
    }

    GpuEvent ExecutionContext::ResourceBarrier(
        absl::Span<const D3D12_RESOURCE_BARRIER> barriers)
    {
        std::unique_lock<std::mutex> lock(batch_state_->mutex);

        // The caller may not keep the barriers referenced by the span alive for
        // longer than this function call, so make a copy and transfer ownership to
        // the lambda.
        absl::InlinedVector<D3D12_RESOURCE_BARRIER, 4> barriers_copy(
            barriers.begin(),
            barriers.end());
        batch_state_->WriteBatch().emplace_back(
            [=, barriers = std::move(barriers_copy)](DmlCommandList& command_list)
            { command_list.ResourceBarrier(barriers); });

        batch_state_->command_added.notify_all();

        return batch_state_->next_flush_event;
    }

    GpuEvent ExecutionContext::UavBarrier()
    {
        std::unique_lock<std::mutex> lock(batch_state_->mutex);

        batch_state_->WriteBatch().emplace_back([=](DmlCommandList& command_list)
                                                { command_list.UavBarrier(); });

        batch_state_->command_added.notify_all();

        return batch_state_->next_flush_event;
    }

    GpuEvent ExecutionContext::Flush()
    {
        std::unique_lock<std::mutex> lock(batch_state_->mutex);
        auto event = batch_state_->next_flush_event;
        if (batch_state_->WriteBatch().empty())
        {
            --event.fenceValue;
        }

        batch_state_->flush_requested = true;
        batch_state_->command_added.notify_all();
        return event;
    }

    Status ExecutionContext::GetCommandRecorderStatus() const
    {
        std::unique_lock<std::mutex> lock(batch_state_->mutex);
        return batch_state_->status;
    }

    GpuEvent ExecutionContext::GetCurrentCompletionEvent()
    {
        std::unique_lock<std::mutex> lock(batch_state_->mutex);
        auto event = batch_state_->next_flush_event;
        if (batch_state_->WriteBatch().empty())
        {
            --event.fenceValue;
        }
        return event;
    }

    D3D12_COMMAND_LIST_TYPE ExecutionContext::GetCommandListTypeForQueue() const
    {
        // No need to acquire the lock since the queue type is immutable once the
        // queue is constructed.
        return dml_command_queue_->GetType();
    }

    void ExecutionContext::QueueReference(IUnknown* object)
    {
        dml_command_queue_->QueueReference(object, true);
    }

    void ExecutionContext::ReleaseCompletedReferences()
    {
        dml_command_queue_->ReleaseCompletedReferences();
    }

    void ExecutionContext::GetCommandListForRecordingAndInvalidateState(ID3D12GraphicsCommandList** commandList)
    {
        // Ensure the descriptor heap is reset to D3D as something external may change it before recording
        dml_command_list_->InvalidateDescriptorHeap();
        dml_command_list_->GetCommandList().CopyTo(commandList);
    }

    /*static*/ void ExecutionContext::ExecutionThreadProc(
        std::shared_ptr<BatchState> state,
        std::shared_ptr<DmlCommandList> command_list,
        std::shared_ptr<CommandQueue> command_queue,
        uint32_t batch_flush_size,
        uint32_t batch_flush_time_us)
    {
        auto last_flush_time = std::chrono::steady_clock::now();

        while (true)
        {
            std::chrono::duration<double> elapsed =
                std::chrono::steady_clock::now() - last_flush_time;
            auto elapsed_us = elapsed.count() * 1e6;

            std::unique_lock<std::mutex> lock(state->mutex);
            if (state->exit_requested)
            {
                break;
            }

            auto& batch = state->WriteBatch();

            if (batch.empty())
            {
                // Wait for new work to be batched.
                state->command_added.wait(lock);

                // Return to the top in case of spurious wakeup.
                continue;
            }

            // Check if it's time to swap the write/execute batches and flush work
            // to the GPU: this occurs if a flush is explicitly requested, the batch
            // has reached a certain size, or enough time has elapsed since the last
            // flush. The goal here is to balance feeding the GPU work while the CPU
            // is processing more commands and avoiding many small packets.
            bool flush = false;
            if (state->flush_requested || batch.size() >= batch_flush_size ||
                elapsed_us >= batch_flush_time_us)
            {
                state->write_batch_index = (state->write_batch_index + 1) % 2;
                flush = true;
                ++state->next_flush_event.fenceValue;
            }
            state->flush_requested = false;

            // Unlock to allow kernels to resume writing to the new write batch.
            lock.unlock();

            if (flush)
            {
                auto status = RecordAndExecute(command_queue.get(), command_list.get(), batch);

                if (!status.IsOK())
                {
                    lock.lock();
                    state->status = status;
                    lock.unlock();
                    break;
                }

                last_flush_time = std::chrono::steady_clock::now();
            }
        }
    }

    /*static*/ Status ExecutionContext::RecordAndExecute(
        CommandQueue* command_queue,
        DmlCommandList* command_list,
        Batch& batch)
    {
        // Record the commands into the command list.
        command_list->Open();
        for (auto& command : batch)
        {
            command(*command_list);
        }
        auto status = command_list->Close();

        if (!status.IsOK())
        {
            return status;
        }

        ID3D12CommandList* command_lists[] = {command_list->Get()};
        command_queue->ExecuteCommandLists(command_lists);
        command_queue->ReleaseCompletedReferences();
        batch.clear();

        return Status::OK();
    }

} // namespace Dml
