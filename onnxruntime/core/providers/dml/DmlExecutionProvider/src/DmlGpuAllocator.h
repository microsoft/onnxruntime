// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "DmlBufferRegion.h"
#include "DmlBuffer.h"

namespace Dml
{
    class DmlReservedResourceSubAllocator;
    class AllocationInfo;
    struct TaggedPointer;

    class DmlGpuAllocator : public onnxruntime::IAllocator
    {
    public:
        DmlGpuAllocator(onnxruntime::IAllocator* bfcAllocator, std::shared_ptr<DmlReservedResourceSubAllocator> subAllocator);

        void* Alloc(size_t size_in_bytes) final;
        void Free(void* ptr) final;
        D3D12BufferRegion CreateBufferRegion(const TaggedPointer& taggedPointer, uint64_t size_in_bytes);
        AllocationInfo* GetAllocationInfo(const TaggedPointer& taggedPointer);
        void SetDefaultRoundingMode(AllocatorRoundingMode roundingMode);
        DmlBuffer AllocateDefaultBuffer(uint64_t num_bytes);

    private:
        // This allocator is managed by ORT and should be used to allocate/free memory in order
        // to utilize the BFC acapabilities
        onnxruntime::IAllocator* m_bfcAllocator;

        // This allocator is specific to DML and is used to decode the opaque data returned by the BFC
        // allocator into objects that DML understands
        std::shared_ptr<DmlReservedResourceSubAllocator> m_subAllocator;
    };
} // namespace Dml
