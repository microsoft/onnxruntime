// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "core/session/onnxruntime_c_api.h"

#include "BucketizedBufferAllocator.h"
#include "DmlSubAllocator.h"
#include "ReadbackHeap.h"
#include "PooledUploadHeap.h"
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
        std::shared_ptr<ExecutionContext> context,
        const D3D12_HEAP_PROPERTIES& heapProps,
        D3D12_HEAP_FLAGS heapFlags,
        D3D12_RESOURCE_FLAGS resourceFlags,
        D3D12_RESOURCE_STATES initialState,
        std::unique_ptr<DmlSubAllocator>&& subAllocator
        )
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
        m_context(context),
        m_subAllocator(std::move(subAllocator))
    {
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
        return Alloc(size, m_defaultRoundingMode);
    }

    void* BucketizedBufferAllocator::Alloc(size_t size, AllocatorRoundingMode roundingMode)
    {
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

        assert(resourceWrapper != nullptr);
        assert(resourceWrapper->GetD3D12Resource()->GetDesc().Width == bucketSize);

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

    void BucketizedBufferAllocator::FreeResource(void* p, uint64_t pooledResourceId)
    {
        AllocationInfo *allocInfo = static_cast<AllocationInfo*>(p);

        assert(allocInfo != nullptr); // Can't free nullptr

        if (allocInfo->GetOwner() != this)
        {
            // This allocation doesn't belong to this allocator!
            ORT_THROW_HR(E_INVALIDARG);
        }

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
            // Free the underlying allocation once queued work has completed.
#ifdef _GAMING_XBOX
            m_context->QueueReference(WRAP_GRAPHICS_UNKNOWN(allocInfo->GetResource()).Get());
#else
            m_context->QueueReference(allocInfo->GetResource());
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
        return allocInfo;
    }

    void BucketizedBufferAllocator::SetDefaultRoundingMode(AllocatorRoundingMode roundingMode)
    {
        m_defaultRoundingMode = roundingMode;
    }

    void BucketizedBufferAllocator::SetResidency(bool value)
    {
      //Check if we need to change residency
      if (m_isResident == value) return;
      m_isResident = value;

      //Collect all pageables
      std::vector<ID3D12Pageable*> pageables;
      std::vector<D3D12_RESIDENCY_PRIORITY> priorities;

      pageables.reserve(m_pool.size());
      priorities.reserve(m_pool.size());

      for (auto& bucket : m_pool)
      {
        for (auto& resource : bucket.resources)
        {
          auto d3d12Resource = resource.resource.Get()->GetD3D12Resource();

          ID3D12Pageable* d3d12Pageable = nullptr;
          d3d12Resource->QueryInterface<ID3D12Pageable>(&d3d12Pageable);

          if (d3d12Pageable)
          {
            pageables.push_back(d3d12Pageable);
            priorities.push_back(value ? D3D12_RESIDENCY_PRIORITY_MAXIMUM : D3D12_RESIDENCY_PRIORITY_MINIMUM);
          }
        }
      }

      //Set residency
      if (value)
      {
        ORT_THROW_IF_FAILED(m_device->MakeResident(uint32_t(pageables.size()), pageables.data()));
      }
      else
      {
        ORT_THROW_IF_FAILED(m_device->Evict(uint32_t(pageables.size()), pageables.data()));
      }

      //Set residency priority      
      ID3D12Device1* d3d12Device1 = nullptr;
      if(m_device->QueryInterface<ID3D12Device1>(&d3d12Device1))
      {
        ORT_THROW_IF_FAILED(d3d12Device1->SetResidencyPriority(uint32_t(pageables.size()), pageables.data(), priorities.data()));
      }

      //Release handles
      for (auto pageable : pageables)
      {
        pageable->Release();
      }
    }

    void BucketizedBufferAllocator::SetResidency2(bool value)
    {
      m_subAllocator->SetResidency(value);
    }

    void BucketizedBufferAllocator::PageOutAll(ReadbackHeap * readbackHeap)
    {
      //Enumerate all buckets
      for (auto& bucket : m_pool)
      {
        //Enumerate resources in bucket
        for (auto& item : bucket.resources)
        {
          //Get resource
          auto resource = item.resource->GetD3D12Resource();

          //Determine size
          auto resourceDesc = resource->GetDesc();
          item.resourceBackup.resize(resourceDesc.Width);

          //Copy data from GPU to CPU
          readbackHeap->ReadbackFromGpu(
            item.resourceBackup,
            resource,
            0,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

          //Release GPU memory
          //m_context->QueueReference(resource);
          item.resource.Reset();
        }
      }
    }

    void BucketizedBufferAllocator::PageInAll(PooledUploadHeap * uploadHeap)
    {
      //Enumerate all buckets
      for (auto& bucket : m_pool)
      {
        //Enumerate resources in bucket
        for (auto& item : bucket.resources)
        {
          //Allocate GPU memory
          item.resource = m_subAllocator->Alloc(item.resourceBackup.size());

          //Get resource
          auto resource = item.resource->GetD3D12Resource();

          //Copy data from CPU to GPU
          uploadHeap->BeginUploadToGpu(
            resource,
            0,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            item.resourceBackup);
          
          //Release CPU memory
          item.resourceBackup.clear();
        }
      }
    }

    CPUAllocator::CPUAllocator(OrtMemType memType)
        : onnxruntime::IAllocator(
            OrtMemoryInfo(
                "DML CPU",
                OrtAllocatorType::OrtDeviceAllocator,
                OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0),
                0,
                memType
            )
        )
    {
    }

    void* CPUAllocator::Alloc(size_t size)
    {
        return onnxruntime::AllocatorDefaultAlloc(size);
    }

    void CPUAllocator::Free(void* p)
    {
        return onnxruntime::AllocatorDefaultFree(p);
    }

} // namespace Dml
