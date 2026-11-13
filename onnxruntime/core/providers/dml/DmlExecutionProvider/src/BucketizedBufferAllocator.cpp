// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include <bit>

#include "core/session/onnxruntime_c_api.h"

#include "BucketizedBufferAllocator.h"
#include "DmlSubAllocator.h"
// #define PRINT_OUTSTANDING_ALLOCATIONS

namespace Dml
{
    AllocationInfo::~AllocationInfo()
    {
        if (m_owner)
        {
            m_owner->FreeResource(this, m_pooledResourceId);
        }
    }

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
        ExecutionContext* context,
        const D3D12_HEAP_PROPERTIES& heapProps,
        D3D12_HEAP_FLAGS heapFlags,
        D3D12_RESOURCE_FLAGS resourceFlags,
        D3D12_RESOURCE_STATES initialState,
        std::unique_ptr<DmlSubAllocator>&& subAllocator)
        : onnxruntime::IAllocator(
              OrtMemoryInfo(
                  "DML",
                  OrtAllocatorType::OrtDeviceAllocator,
                  OrtDevice(OrtDevice::DML, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::MICROSOFT, 0))),
          m_device(device),
          m_heapProperties(heapProps),
          m_heapFlags(heapFlags),
          m_resourceFlags(resourceFlags),
          m_initialState(initialState),
          m_context(context),
          m_subAllocator(std::move(subAllocator)) {
    }

    /*static*/ gsl::index BucketizedBufferAllocator::GetBucketIndexFromSize(uint64_t size)
    {
        assert(size != 0 && size <= c_maxBucketSize);

        // Each bucket is twice as large as the previous one, in ascending order. This is ceil(log2(size)) computed
        // with integers: floating-point log2 rounds sizes slightly above 2^n (n >= 49) down to n.
        gsl::index index = static_cast<gsl::index>(std::bit_width(size - 1));
        assert((1ull << index) >= size);

        // The smallest bucket is 2^n bytes large, where n = c_minResourceSizeExponent
        index = std::max<gsl::index>(index, c_minResourceSizeExponent);
        index -= c_minResourceSizeExponent;

        return index;
    }

    /*static*/ uint64_t BucketizedBufferAllocator::GetBucketSizeFromIndex(gsl::index index)
    {
        return (1ull << (index + c_minResourceSizeExponent));
    }

    void* BucketizedBufferAllocator::Alloc(size_t size)
    {
        return Alloc(size, m_defaultRoundingMode);
    }

    void* BucketizedBufferAllocator::Alloc(size_t size, AllocatorRoundingMode roundingMode)
    {
        // For some reason lotus likes requesting 0 bytes of memory
        size = std::max<size_t>(1, size);

        // Larger sizes can't be rounded up to a bucket size without overflowing, and no device could allocate them.
        if (size > c_maxBucketSize)
        {
            ORT_THROW_HR(E_OUTOFMEMORY);
        }

        ComPtr<DmlResourceWrapper> resourceWrapper;
        uint64_t resourceId = 0;
        uint64_t bucketSize = 0;
        std::optional<size_t> poolBucketIndex;

        // Resources which aren't pooled are created with their size rounded up to a multiple of 4 bytes.
        const uint64_t unpooledSize = (static_cast<uint64_t>(size) + 3) & ~uint64_t{3};

        // Use a pooled resource if rounding is requested, or if the resource would be exactly a bucket size anyway.
        // The latter must compare the size of the resource that would be created rather than the requested size,
        // since FreeResource returns every resource allocated from the pool back to its bucket.
        if (roundingMode == AllocatorRoundingMode::Enabled || unpooledSize == GetBucketSizeFromIndex(GetBucketIndexFromSize(size)))
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
                resourceWrapper = m_subAllocator->Alloc(onnxruntime::narrow<size_t>(bucketSize));
                resourceId = ++m_currentResourceId;
            }
            else
            {
                // Retrieve a resource from the bucket
                resourceWrapper = std::move(bucket->resources.back().resource);
                resourceId = bucket->resources.back().resourceId;
                bucket->resources.pop_back();
            }

            poolBucketIndex = gsl::narrow_cast<size_t>(bucketIndex);
        }
        else
        {
            // The allocation will not be pooled.  Construct a new one
            bucketSize = unpooledSize;
            resourceWrapper = m_subAllocator->Alloc(onnxruntime::narrow<size_t>(bucketSize));
            resourceId = ++m_currentResourceId;
        }

        assert(resourceWrapper->GetD3D12Resource()->GetDesc().Width == bucketSize);
        assert(resourceWrapper != nullptr);

        ComPtr<AllocationInfo> allocInfo = Dml::SafeMakeOrThrow<AllocationInfo>(
            this,
            ++m_currentAllocationId,
            resourceId,
            resourceWrapper.Get(),
            size,
            poolBucketIndex
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

    void BucketizedBufferAllocator::FreeResource(void* p, uint64_t pooledResourceId)
    {
        AllocationInfo *allocInfo = static_cast<AllocationInfo*>(p);

        assert(allocInfo != nullptr); // Can't free nullptr

        if (allocInfo->GetOwner() != this)
        {
            // This allocation doesn't belong to this allocator!
            ORT_THROW_HR(E_INVALIDARG);
        }

        // Return the resource to the pool only if it was allocated from the pool. Its bucket was recorded by Alloc,
        // which also grew the pool to contain that bucket, so the remaining checks always hold. They are kept as
        // defense in depth: any other resource is released rather than written past the end of the pool or into a
        // bucket of the wrong size.
        const std::optional<size_t> bucketIndex = allocInfo->GetPoolBucketIndex();
        const bool returnToPool =
            bucketIndex.has_value() &&
            *bucketIndex < m_pool.size() &&
            GetBucketSizeFromIndex(gsl::narrow_cast<gsl::index>(*bucketIndex)) == allocInfo->GetResource()->GetDesc().Width;
        assert(returnToPool == bucketIndex.has_value());

        if (returnToPool)
        {
            // Return the resource to the bucket
            Bucket* bucket = &m_pool[*bucketIndex];

            Resource resource = {allocInfo->DetachResourceWrapper(), pooledResourceId};
            bucket->resources.push_back(resource);
        }
        else
        {
            if (!m_context->IsClosed())
            {
                // Free the underlying allocation once queued work has completed.
    #ifdef _GAMING_XBOX
                m_context->QueueReference(WRAP_GRAPHICS_UNKNOWN(allocInfo->GetResource()).Get());
    #else
                m_context->QueueReference(allocInfo->GetResource());
    #endif
            }

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
        return allocInfo;
    }

    void BucketizedBufferAllocator::SetDefaultRoundingMode(AllocatorRoundingMode roundingMode)
    {
        m_defaultRoundingMode = roundingMode;
    }
} // namespace Dml
