// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

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
        assert(size != 0);

        // Each bucket is twice as large as the previous one, in ascending order
        gsl::index index = static_cast<gsl::index>(ceil(log2(size)));
        assert((1ull << index) >= size); // This must be true unless there were some strange rounding issues

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
        return Alloc(size, m_defaultRoundingMode.load(std::memory_order_acquire));
    }

    void* BucketizedBufferAllocator::Alloc(size_t size, AllocatorRoundingMode roundingMode)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        // For some reason lotus likes requesting 0 bytes of memory
        size = std::max<size_t>(1, size);

        ComPtr<DmlResourceWrapper> resourceWrapper;
        uint64_t resourceId = 0;
        uint64_t bucketSize = 0;

        // Use a pooled resource if the size (post rounding, if requested) matches a bucket size
        if (roundingMode == AllocatorRoundingMode::Enabled || size == GetBucketSizeFromIndex(GetBucketIndexFromSize(size)))
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
        }
        else
        {
            // The allocation will not be pooled.  Construct a new one
            bucketSize = (size + 3) & ~3;
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
            size
        );

    #if _DEBUG
        m_outstandingAllocationsById[allocInfo->GetId()] = allocInfo.Get();
    #endif

        return allocInfo.Detach();
    }

    void BucketizedBufferAllocator::Free(void* p)
    {
        // No lock here: the ComPtr release may trigger AllocationInfo::~AllocationInfo
        // which calls FreeResource() — that method acquires m_mutex itself.
        // COM ref-count operations are already interlocked.
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

        // Capture resource outside lock to avoid lock-order inversion:
        // allocator → context vs context → queue → destructor → allocator
        ComPtr<DmlResourceWrapper> detachedWrapper;
        bool needsQueueReference = false;

        {
            std::lock_guard<std::mutex> lock(m_mutex);

            // Free the resource to the pool if its size matches a bucket size
            gsl::index bucketIndex = GetBucketIndexFromSize(allocInfo->GetRequestedSize());
            if (GetBucketSizeFromIndex(bucketIndex) == allocInfo->GetResource()->GetDesc().Width)
            {
                assert(gsl::narrow_cast<gsl::index>(m_pool.size()) > bucketIndex);

                // Return the resource to the bucket
                Bucket* bucket = &m_pool[bucketIndex];

                Resource resource = {allocInfo->DetachResourceWrapper(), pooledResourceId};
                bucket->resources.push_back(resource);
            }
            else
            {
                detachedWrapper = allocInfo->DetachResourceWrapper();
                needsQueueReference = true;
            }

        #if _DEBUG
            assert(m_outstandingAllocationsById[allocInfo->GetId()] == allocInfo);
            m_outstandingAllocationsById.erase(allocInfo->GetId());
        #endif
        }

        // Call into ExecutionContext OUTSIDE the allocator lock to prevent
        // lock-order inversion (allocator→context vs context→queue→allocator)
        if (needsQueueReference && !m_context->IsClosed())
        {
            // Free the underlying allocation once queued work has completed.
    #ifdef _GAMING_XBOX
            m_context->QueueReference(WRAP_GRAPHICS_UNKNOWN(detachedWrapper->GetD3D12Resource()).Get());
    #else
            m_context->QueueReference(detachedWrapper->GetD3D12Resource());
    #endif
        }
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
        m_defaultRoundingMode.store(roundingMode, std::memory_order_release);
    }
} // namespace Dml
