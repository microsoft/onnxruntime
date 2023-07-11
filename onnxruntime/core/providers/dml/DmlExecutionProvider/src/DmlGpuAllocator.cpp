// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "precomp.h"
#include "DmlGpuAllocator.h"
#include "core/framework/allocator.h"
#include "DmlReservedResourceSubAllocator.h"
#include "DmlTaggedPointer.h"

namespace Dml
{
    DmlGpuAllocator::DmlGpuAllocator(onnxruntime::IAllocator* bfcAllocator, std::shared_ptr<DmlReservedResourceSubAllocator> subAllocator)
    : onnxruntime::IAllocator(
        OrtMemoryInfo(
            onnxruntime::DML,
            OrtAllocatorType::OrtDeviceAllocator,
            OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0),
            0
        )
    ),
    m_bfcAllocator(bfcAllocator),
    m_subAllocator(subAllocator) {}

    void* DmlGpuAllocator::Alloc(size_t size_in_bytes)
    {
        return m_bfcAllocator->Alloc(size_in_bytes);
    }

    void DmlGpuAllocator::Free(void* ptr)
    {
        m_bfcAllocator->Free(ptr);
    }

    D3D12BufferRegion DmlGpuAllocator::CreateBufferRegion(const TaggedPointer& taggedPointer, uint64_t size_in_bytes)
    {
        return m_subAllocator->CreateBufferRegion(taggedPointer, size_in_bytes);
    }

    AllocationInfo* DmlGpuAllocator::GetAllocationInfo(const TaggedPointer& taggedPointer)
    {
        return m_subAllocator->GetAllocationInfo(taggedPointer);
    }

    void DmlGpuAllocator::SetDefaultRoundingMode(AllocatorRoundingMode roundingMode)
    {
        m_subAllocator->SetDefaultRoundingMode(roundingMode);
    }

    DmlBuffer DmlGpuAllocator::AllocateDefaultBuffer(uint64_t num_bytes)
    {
        return DmlBuffer(this, num_bytes);
    }

} // namespace Dml
