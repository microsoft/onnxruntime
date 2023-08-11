// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "DmlBufferRegion.h"
#include "DmlBuffer.h"
#include "DmlAllocatorRoundingMode.h"

namespace onnxruntime
{
    class BFCArena;
}

namespace Dml
{
    class DmlReservedResourceSubAllocator;
    class BucketizedBufferAllocator;
    class AllocationInfo;
    struct TaggedPointer;

    enum class ActiveAllocator
    {
        BfcAllocator,
        BucketizedBufferAllocator,
    };

    class DmlGpuAllocator : public onnxruntime::IAllocator
    {
    public:
        DmlGpuAllocator(
            onnxruntime::BFCArena* bfcAllocator,
            BucketizedBufferAllocator* bucketizedBufferAllocator,
            std::shared_ptr<DmlReservedResourceSubAllocator> bfcSubAllocator,
            ActiveAllocator activeAllocator);

        void* Alloc(size_t sizeInBytes) final;
        void* Alloc(size_t sizeInBytes, AllocatorRoundingMode roundingMode);
        void Free(void* ptr) final;
        D3D12BufferRegion CreateBufferRegion(void* opaquePointer, uint64_t sizeInBytes);
        AllocationInfo* GetAllocationInfo(void* opaquePointer);
        void SetDefaultRoundingMode(AllocatorRoundingMode roundingMode);
        DmlBuffer AllocateDefaultBuffer(uint64_t num_bytes);
        DmlBuffer AllocateDefaultBuffer(uint64_t num_bytes, AllocatorRoundingMode roundingMode);
        uint64_t GetUniqueId(void* opaquePointer);

    private:
        // This allocator is managed by ORT and should be used to allocate/free memory in order
        // to utilize the BFC acapabilities
        onnxruntime::BFCArena* m_bfcAllocator;

        // This allocator is the old bucketized allocator that is kept for backward compatibility purposes
        // and is only used when external custom ops are registered.
        BucketizedBufferAllocator* m_bucketizedBufferAllocator;

        // This allocator is specific to DML and is used to decode the opaque data returned by the BFC
        // allocator into objects that DML understands
        std::shared_ptr<DmlReservedResourceSubAllocator> m_bfcSubAllocator;

        // Unless specifically requested, allocation sizes are not rounded to enable pooling
        // until SetDefaultRoundingMode is called.  This should be done at completion of session
        // initialization.
        AllocatorRoundingMode m_defaultRoundingMode = AllocatorRoundingMode::Disabled;

        ActiveAllocator m_activeAllocator;
    };
} // namespace Dml
