// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "core/session/onnxruntime_c_api.h"

#include "BucketizedBufferAllocator.h"
#include "DmlAllocationInfo.h"
#include "DmlCommittedResourceWrapper.h"
// #define PRINT_OUTSTANDING_ALLOCATIONS

namespace Dml
{
    BucketizedBufferAllocator::~BucketizedBufferAllocator()
    {
#ifdef PRINT_OUTSTANDING_ALLOCATIONS
        if (!m_outstandingAllocationsById.empty())
        {
            printf("BucketizedBufferAllocator outstanding allocation indices:\n");
            for (auto& entry : m_outstandingAllocationsById)
            {
                printf("%u\n", static_cast<int>(entry.first));
            }
            printf("\n");
        }
#endif
    }

    BucketizedBufferAllocator::BucketizedBufferAllocator(
        ID3D12Device* device,
        std::shared_ptr<ExecutionContext> context,
        const D3D12_HEAP_PROPERTIES& heapProps,
        D3D12_HEAP_FLAGS heapFlags,
        D3D12_RESOURCE_FLAGS resourceFlags,
        D3D12_RESOURCE_STATES initialState)
        : onnxruntime::IAllocator(
            OrtMemoryInfo(
                "DML",
                OrtAllocatorType::OrtDeviceAllocator,
                OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)
            )
        ),
        m_device(device),
        m_heapProperties(heapProps),
        m_heapFlags(heapFlags),
        m_resourceFlags(resourceFlags),
        m_initialState(initialState),
        m_context(context)
    {
    }

    /*static*/ gsl::index BucketizedBufferAllocator::GetBucketIndexFromSize(uint64_t size)
    {
        assert(size != 0);

        // Each bucket is twice as large as the previous one, in ascending order
        gsl::index index = static_cast<gsl::index>(ceil(log2(size)));
        assert((1ull << index) >= size); // This must be true unless there were some strange rounding issues

        // The smallest bucket is 2^n bytes large, where n = MinResourceSizeExponent
        index = std::max<gsl::index>(index, MinResourceSizeExponent);
        index -= MinResourceSizeExponent;

        return index;
    }

    /*static*/ uint64_t BucketizedBufferAllocator::GetBucketSizeFromIndex(gsl::index index)
    {
        return (1ull << (index + MinResourceSizeExponent));
    }

    ComPtr<DmlResourceWrapper> BucketizedBufferAllocator::AllocCommittedResource(size_t size)
    {
        ComPtr<ID3D12Resource> resource;
        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ORT_THROW_IF_FAILED(m_device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &buffer,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(resource.GetAddressOf())
        ));

        ComPtr<DmlResourceWrapper> resourceWrapper;
        wil::MakeOrThrow<DmlCommittedResourceWrapper>(std::move(resource)).As(&resourceWrapper);
        return resourceWrapper;
    }

    AllocationInfo* BucketizedBufferAllocator::GetAllocationInfo(void* opaquePointer)
    {
        return static_cast<AllocationInfo*>(opaquePointer);
    }

    D3D12BufferRegion BucketizedBufferAllocator::CreateBufferRegion(void* opaquePointer, uint64_t sizeInBytes) const
    {
        auto allocationInfo = static_cast<AllocationInfo*>(opaquePointer);

        // Make sure that we are aligned to 4 bytes to satisfy DML's requirements
        constexpr uint64_t DML_ALIGNMENT = 4;
        sizeInBytes = (1 + (sizeInBytes - 1) / DML_ALIGNMENT) * DML_ALIGNMENT;

        return D3D12BufferRegion(0, sizeInBytes, allocationInfo->GetD3D12Resource());
    }

    void* BucketizedBufferAllocator::Alloc(size_t size)
    {
        // For some reason lotus likes requesting 0 bytes of memory
        size = std::max<size_t>(1, size);

        ComPtr<DmlResourceWrapper> resourceWrapper;
        uint64_t resourceId = 0;
        uint64_t bucketSize = 0;

        // Use a pooled resource if the size (post rounding, if requested) matches a bucket size
        if (m_defaultRoundingMode == AllocatorRoundingMode::Enabled || size == GetBucketSizeFromIndex(GetBucketIndexFromSize(size)))
        {
            Bucket* bucket = nullptr;

            // Find the bucket for this allocation size
            gsl::index bucketIndex = GetBucketIndexFromSize(size);

            if (gsl::narrow_cast<gsl::index>(m_pool.size()) <= bucketIndex)
            {
                // Ensure there are sufficient buckets
                m_pool.resize(bucketIndex + 1);
            }

            bucket = &m_pool[bucketIndex];
            bucketSize = GetBucketSizeFromIndex(bucketIndex);

            if (bucket->resources.empty())
            {
                // No more resources in this bucket - allocate a new one
                resourceWrapper = AllocCommittedResource(onnxruntime::narrow<size_t>(bucketSize));
                resourceId = ++m_currentResourceId;
            }
            else
            {
                // Retrieve a resource from the bucket
                resourceWrapper = std::move(bucket->resources.back().resource);
                resourceId = bucket->resources.back().resourceId;
                bucket->resources.pop_back();
            }
        }
        else
        {
            // The allocation will not be pooled.  Construct a new one
            bucketSize = (size + 3) & ~3;
            resourceWrapper = AllocCommittedResource(onnxruntime::narrow<size_t>(bucketSize));
            resourceId = ++m_currentResourceId;
        }

        assert(resourceWrapper->GetD3D12Resource()->GetDesc().Width == bucketSize);
        assert(resourceWrapper != nullptr);

        ComPtr<AllocationInfo> allocInfo = wil::MakeOrThrow<AllocationInfo>(
            this,
            ++m_currentAllocationId,
            resourceId,
            resourceWrapper.Get(),
            size
        );

    #if _DEBUG
        m_outstandingAllocationsById[allocInfo->GetId()] = allocInfo.Get();
    #endif

        return allocInfo.Detach();
    }

    void BucketizedBufferAllocator::Free(void* p)
    {
        // Release Lotus's reference on the allocation.  The allocation
        // also inherits IUnknown, and once its final reference reaches zero
        // it will call FreeResource
        ComPtr<AllocationInfo> allocInfo;
        allocInfo.Attach(static_cast<AllocationInfo*>(p));
    }

    uint64_t BucketizedBufferAllocator::GetUniqueId(void* opaquePointer)
    {
        const auto* allocInfo = static_cast<const AllocationInfo*>(opaquePointer);
        return allocInfo->GetPooledResourceId();
    }

    void BucketizedBufferAllocator::FreeResource(AllocationInfo* allocInfo, uint64_t pooledResourceId)
    {
        assert(allocInfo != nullptr); // Can't free nullptr

        if (allocInfo->GetOwner() != this)
        {
            // This allocation doesn't belong to this allocator!
            ORT_THROW_HR(E_INVALIDARG);
        }

        // Free the resource to the pool if its size matches a bucket size
        gsl::index bucketIndex = GetBucketIndexFromSize(allocInfo->GetRequestedSize());
        if (GetBucketSizeFromIndex(bucketIndex) == allocInfo->GetD3D12Resource()->GetDesc().Width)
        {
            assert(gsl::narrow_cast<gsl::index>(m_pool.size()) > bucketIndex);

            // Return the resource to the bucket
            Bucket* bucket = &m_pool[bucketIndex];

            Resource resource = {allocInfo->DetachResourceWrapper(), pooledResourceId};
            bucket->resources.push_back(resource);
        }
        else
        {
            // Free the underlying allocation once queued work has completed.
#ifdef _GAMING_XBOX
            m_context->QueueReference(WRAP_GRAPHICS_UNKNOWN(allocInfo->GetD3D12Resource()).Get());
#else
            m_context->QueueReference(allocInfo->GetD3D12Resource());
#endif
            allocInfo->DetachResourceWrapper();
        }

    #if _DEBUG
        assert(m_outstandingAllocationsById[allocInfo->GetId()] == allocInfo);
        m_outstandingAllocationsById.erase(allocInfo->GetId());
    #endif

        // The allocation info is already destructing at this point
    }


    const AllocationInfo* BucketizedBufferAllocator::DecodeDataHandle(const void* opaqueHandle)
    {
        if (opaqueHandle == nullptr)
        {
            // There is no memory allocated which needs to be decoded.
            ORT_THROW_HR(E_INVALIDARG);
        }
        const auto* allocInfo = static_cast<const AllocationInfo*>(opaqueHandle);

        auto owner = allocInfo->GetOwner();
        //The owner can be null if the resource was wrapped via CreateGPUAllocationFromD3DResource
        if (owner != nullptr && owner != this)
        {
            // This allocation doesn't belong to this allocator!
            ORT_THROW_HR(E_INVALIDARG);
        }

        return allocInfo;
    }

    void BucketizedBufferAllocator::SetDefaultRoundingMode(AllocatorRoundingMode roundingMode)
    {
        m_defaultRoundingMode = roundingMode;
    }

} // namespace Dml
