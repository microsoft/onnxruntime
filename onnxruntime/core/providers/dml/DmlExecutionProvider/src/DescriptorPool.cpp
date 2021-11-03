// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
    DescriptorHeap::DescriptorHeap(ID3D12DescriptorHeap* heap) :
        m_heap(heap),
        m_capacity(heap->GetDesc().NumDescriptors),
        m_headCpuHandle(heap->GetCPUDescriptorHandleForHeapStart()),
        m_headGpuHandle(heap->GetGPUDescriptorHandleForHeapStart()),
        m_heapFlags(heap->GetDesc().Flags)
    {
        ComPtr<ID3D12Device> device;
        ORT_THROW_IF_FAILED(heap->GetDevice(IID_PPV_ARGS(&device)));

        m_handleIncrementSize = device->GetDescriptorHandleIncrementSize(heap->GetDesc().Type);
    }

    std::optional<DescriptorRange> DescriptorHeap::TryAllocDescriptors(
        uint32_t numDescriptors, 
        GpuEvent completionEvent,
        D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags
        )
    {
        // Bail if the desired heap creation flags are incompatible with the existing heap.
        if (m_heapFlags != heapFlags)
        {
            return std::nullopt;
        }

        if ((m_completionEvent.fence != nullptr) && (m_completionEvent.IsSignaled()))
        {
            // This class always allocates descriptors from the end of the heap.
            // If the most recent completion event is signaled, then all previous
            // allocations have completed; the entire capacity is available to use.
            m_size = 0;
            m_headCpuHandle = m_heap->GetCPUDescriptorHandleForHeapStart();
            m_headGpuHandle = m_heap->GetGPUDescriptorHandleForHeapStart();
        }

        // The caller will need to create a new heap if there is no space left in this one.
        uint32_t spaceRemaining = m_capacity - m_size;
        if (spaceRemaining < numDescriptors)
        {
            return std::nullopt;
        }

        DescriptorRange range = { m_heap.Get(), m_headCpuHandle, m_headGpuHandle };

        m_size += numDescriptors;
        m_completionEvent = completionEvent;
        m_headCpuHandle.Offset(numDescriptors, m_handleIncrementSize);
        m_headGpuHandle.Offset(numDescriptors, m_handleIncrementSize);

        return range;
    }


    DescriptorPool::DescriptorPool(ID3D12Device* device, uint32_t initialCapacity) : 
        m_device(device),
        m_initialHeapCapacity(initialCapacity)
    {
        CreateHeap(initialCapacity, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
    }

    DescriptorRange DescriptorPool::AllocDescriptors(
        uint32_t numDescriptors, 
        GpuEvent completionEvent,
        D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags
        )
    {
        // Attempt to allocate from an existing heap.
        for (DescriptorHeap& heap : m_heaps)
        {
            auto descriptorRange = heap.TryAllocDescriptors(numDescriptors, completionEvent, heapFlags);
            if (descriptorRange.has_value())
            {
                return descriptorRange.value();
            }
        }

        // A new descriptor heap must be created.
        uint32_t newHeapCapacity = std::max(numDescriptors, m_initialHeapCapacity);
        CreateHeap(newHeapCapacity, heapFlags);
        auto descriptorRange = m_heaps.back().TryAllocDescriptors(numDescriptors, completionEvent, heapFlags);
        assert(descriptorRange.has_value());
        return descriptorRange.value();
    }

    void DescriptorPool::Trim()
    {
        // Remove any heaps that are not pending execution.
        auto it = std::remove_if(m_heaps.begin(), m_heaps.end(), [](const DescriptorHeap& heap) {
            auto completionEvent = heap.GetLastCompletionEvent();
            return !completionEvent.fence || completionEvent.IsSignaled();
        });

        m_heaps.erase(it, m_heaps.end());
    }

    void DescriptorPool::CreateHeap(uint32_t numDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS heapFlags)
    {
        // This pool only manages CBV/SRV/UAV descriptors.
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Flags = heapFlags;
        desc.NumDescriptors = numDescriptors;
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

        ComPtr<ID3D12DescriptorHeap> heap;
        ORT_THROW_IF_FAILED(m_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)));

        m_heaps.push_back(DescriptorHeap{heap.Get()});
    }

    uint32_t DescriptorPool::GetTotalCapacity() const
    {
        uint32_t capacity = 0;

        for (auto& heap : m_heaps)
        {
            capacity += heap.GetCapacity();
        }

        return capacity;
    }
}