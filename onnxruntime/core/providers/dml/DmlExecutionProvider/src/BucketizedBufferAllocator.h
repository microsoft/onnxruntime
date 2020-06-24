// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "ExecutionContext.h"

namespace Dml
{
    
    class CPUAllocator : public onnxruntime::IDeviceAllocator
    {
    public:
        explicit CPUAllocator(OrtMemType memType);

        void* Alloc(size_t size) override;
        void Free(void* p) override;
    };

    class BucketizedBufferAllocator;

    class AllocationInfo : public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IUnknown>
    {
    public:
        AllocationInfo(
            BucketizedBufferAllocator* owner,
            size_t id,
            uint64_t pooledResourceId,
            ID3D12Resource* resource,
            size_t requestedSize)
            : m_owner(owner)
            , m_allocationId(id)
            , m_pooledResourceId(pooledResourceId)
            , m_resource(resource)
            , m_requestedSize(requestedSize)
        {}

        ~AllocationInfo();

        BucketizedBufferAllocator* GetOwner() const
        { 
            return m_owner;
        }

        ID3D12Resource* GetResource() const
        { 
            return m_resource.Get();
        }

        ComPtr<ID3D12Resource> DetachResource() const
        { 
            return std::move(m_resource);
        }

        size_t GetRequestedSize() const
        { 
            return m_requestedSize;
        }

        size_t GetId() const
        {
            return m_allocationId;
        } 
        
        uint64_t GetPooledResourceId() const
        {
            return m_pooledResourceId;
        }

    private:
        BucketizedBufferAllocator* m_owner;
        size_t m_allocationId; // For debugging purposes
        uint64_t m_pooledResourceId = 0;
        ComPtr<ID3D12Resource> m_resource;

        // The size requested during Alloc(), which may be smaller than the physical resource size
        size_t m_requestedSize;
    };

    // Implements a Lotus allocator for D3D12 heap buffers, using a bucket allocation strategy. The allocator
    // maintains a set of fixed-size buckets, with each bucket containing one or more D3D12 buffers of that fixed size.
    // All requested allocation sizes are rounded up to the nearest bucket size, which ensures minimal fragmentation
    // while providing an upper bound on the amount of memory "wasted" with each allocation.
    class BucketizedBufferAllocator : public onnxruntime::IAllocator
    {
    public:
        ~BucketizedBufferAllocator();

        // Constructs a BucketizedBufferAllocator which allocates D3D12 committed resources with the specified heap properties,
        // resource flags, and initial resource state.
        BucketizedBufferAllocator(
            ID3D12Device* device,            
            std::shared_ptr<ExecutionContext> context,
            const D3D12_HEAP_PROPERTIES& heapProps,
            D3D12_HEAP_FLAGS heapFlags,
            D3D12_RESOURCE_FLAGS resourceFlags,
            D3D12_RESOURCE_STATES initialState);

        // Returns the information associated with an opaque allocation handle returned by IAllocator::Alloc.
        const AllocationInfo* DecodeDataHandle(const void* opaqueHandle);

        void SetDefaultRoundingMode(AllocatorRoundingMode roundingMode);

    public: // onnxruntime::IAllocator
        void* Alloc(size_t size, AllocatorRoundingMode roundingMode);
        void* Alloc(size_t size) final;
        void Free(void* p) final;

    private:
        static const uint32_t c_minResourceSizeExponent = 16; // 2^16 = 64KB

        // The pool consists of a number of buckets, and each bucket contains a number of resources of the same size.
        // The resources in each bucket are always sized as a power of two, and each bucket contains resources twice
        // as large as the previous bucket.
        struct Resource
        {
            ComPtr<ID3D12Resource> resource;
            uint64_t resourceId;
        };

        struct Bucket
        {
            std::vector<Resource> resources;
        };

        static gsl::index GetBucketIndexFromSize(uint64_t size);
        static uint64_t GetBucketSizeFromIndex(gsl::index index);

        AllocationInfo* DecodeDataHandleInternal(void* opaqueHandle)
        {
            // Implement in terms of const version
            return const_cast<AllocationInfo*>(DecodeDataHandle(static_cast<const void*>(opaqueHandle)));
        }

        friend class AllocationInfo;
        void FreeResource(void* p, uint64_t resourceId);

        ComPtr<ID3D12Device> m_device;
        D3D12_HEAP_PROPERTIES m_heapProperties;
        D3D12_HEAP_FLAGS m_heapFlags;
        D3D12_RESOURCE_FLAGS m_resourceFlags;
        D3D12_RESOURCE_STATES m_initialState;

        std::vector<Bucket> m_pool;
        size_t m_currentAllocationId = 0;
        uint64_t m_currentResourceId = 0;
        AllocatorRoundingMode m_defaultRoundingMode = AllocatorRoundingMode::Enabled;
        std::shared_ptr<ExecutionContext> m_context;

    #if _DEBUG
        // Useful for debugging; keeps track of all allocations that haven't been freed yet
        std::map<size_t, AllocationInfo*> m_outstandingAllocationsById;
    #endif
    };

} // namespace Dml
