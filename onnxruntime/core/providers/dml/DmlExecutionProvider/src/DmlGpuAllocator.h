// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "DmlBufferRegion.h"
#include "DmlManagedBufferRegion.h"

namespace Dml
{
    class BucketizedBufferAllocator;
    class AllocationInfo;

    class DmlGpuAllocator : public onnxruntime::IAllocator
    {
    public:
        DmlGpuAllocator(onnxruntime::IAllocator* bfcAllocator, std::shared_ptr<BucketizedBufferAllocator> subAllocator);

        void* Alloc(size_t size_in_bytes) final;
        void Free(void* ptr) final;
        D3D12BufferRegion CreateBufferRegion(const void* ptr, uint64_t size_in_bytes);
        ComPtr<DmlManagedBufferRegion> CreateManagedBufferRegion(const void* ptr, uint64_t size_in_bytes);
        AllocationInfo* GetAllocationInfo(const void* ptr);
        void SetDefaultRoundingMode(AllocatorRoundingMode roundingMode);

    private:
        // This allocator is managed by ORT and should be used to allocate/free memory in order
        // to utilize the BFC acapabilities
        onnxruntime::IAllocator* m_bfcAllocator;

        // This allocator is specific to DML and is used to decode the opaque data returned by the BFC
        // allocator into objects that DML understands
        std::shared_ptr<BucketizedBufferAllocator> m_subAllocator;
    };
} // namespace Dml
