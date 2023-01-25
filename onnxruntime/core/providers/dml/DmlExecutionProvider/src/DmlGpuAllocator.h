// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "BucketizedBufferAllocator.h"

namespace Dml
{
    class DmlGpuAllocator : public onnxruntime::IAllocator
    {
    public:
        DmlGpuAllocator(onnxruntime::IAllocator* bfcAllocator, std::shared_ptr<BucketizedBufferAllocator> subAllocator)
        : onnxruntime::IAllocator(
            OrtMemoryInfo(
                "DML",
                OrtAllocatorType::OrtDeviceAllocator,
                OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)
            )
        ),
        m_bfcAllocator(bfcAllocator),
        m_subAllocator(std::move(subAllocator)) {}

        void* Alloc(size_t size_in_bytes) final { return m_bfcAllocator->Alloc(size_in_bytes); }
        void Free(void* ptr) final { m_bfcAllocator->Free(ptr); }
        D3D12BufferRegion CreateBufferRegion(const void* ptr, uint64_t size_in_bytes) { return m_subAllocator->CreateBufferRegion(ptr, size_in_bytes); }
        ComPtr<DmlManagedBufferRegion> CreateManagedBufferRegion(const void* ptr, uint64_t size_in_bytes) { return m_subAllocator->CreateManagedBufferRegion(ptr, size_in_bytes); }
        AllocationInfo* GetAllocationInfo(const void* ptr) { return m_subAllocator->GetAllocationInfo(ptr); }
        void SetDefaultRoundingMode(AllocatorRoundingMode roundingMode) { m_subAllocator->SetDefaultRoundingMode(roundingMode); }
        BucketizedBufferAllocator* GetSubAllocator() const { return m_subAllocator.get(); }

    private:
        // This allocator is managed by ORT and should be used to allocate/free memory in order
        // to utilize the BFC acapabilities
        onnxruntime::IAllocator* m_bfcAllocator;

        // This allocator is specific to DML and is used to decode the opaque data returned by the BFC
        // allocator into objects that DML understands
        std::shared_ptr<BucketizedBufferAllocator> m_subAllocator;
    };
} // namespace Dml
