#include "precomp.h"

#include "core/session/onnxruntime_c_api.h"

#include "DmlCommittedResourceWrapper.h"
#include "DmlSubAllocator.h"
#include "TiledBufferAllocator.h"

namespace Dml
{
    TiledBufferAllocator::TiledBufferAllocator(ID3D12Device* device, ExecutionContext* context,
                                               const D3D12_HEAP_PROPERTIES& heapProps, D3D12_HEAP_FLAGS heapFlags,
                                               D3D12_RESOURCE_FLAGS resourceFlags, D3D12_RESOURCE_STATES initialState,
                                               std::unique_ptr<DmlSubAllocator>&& subAllocator) :
        DmlBufferAllocator(OrtMemoryInfo("DML", OrtAllocatorType::OrtDeviceAllocator,
                                         OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0))),
        m_device(device),
        m_heapProperties(heapProps),
        m_heapFlags(heapFlags),
        m_resourceFlags(resourceFlags),
        m_initialState(initialState),
        m_context(context),
        m_pooledAllocator(device, context->Queue()),

#ifdef DML_USE_HEAP_ALLOCATOR_FOR_UNPOOLED_MEMORY
        m_unpooledAllocator(device, context->Queue())
#else
        m_unpooledAllocator(std::move(subAllocator))
#endif
    {
    }

    void TiledBufferAllocator::Clean()
    {
        auto resources = m_pooledAllocator.Clean();
        FreeResources(resources);
    }

    void TiledBufferAllocator::Clear()
    {
        auto resources = m_pooledAllocator.Clear();
        FreeResources(resources);
    }

    DmlAllocatorType TiledBufferAllocator::Type() const
    {
        return DmlAllocatorType::Tiled;
    }

    void* TiledBufferAllocator::Alloc(size_t size, AllocatorPoolingMode poolingMode)
    {
        // For some reason lotus likes requesting 0 bytes of memory
        size = std::max<size_t>(1, size);

        bool isSizePooled = size >= 65536 && !(size & (size - 1));

        // Use a pooled resource if the size (post rounding, if requested) matches a bucket size
        ComPtr<DmlResourceWrapper> resourceWrapper;
        if (poolingMode == AllocatorPoolingMode::Enabled)
        {
            wil::MakeOrThrow<DmlCommittedResourceWrapper>(m_pooledAllocator.AllocateBuffer(size)).As(&resourceWrapper);
        }
        else
        {
#ifdef DML_USE_HEAP_ALLOCATOR_FOR_UNPOOLED_MEMORY
            wil::MakeOrThrow<DmlCommittedResourceWrapper>(m_unpooledAllocator.AllocateBuffer(size))
                .As(&resourceWrapper);
#else
            // The allocation will not be pooled.  Construct a new one
            auto allocationSize = (size + 3) & ~3;
            resourceWrapper = m_unpooledAllocator->Alloc(onnxruntime::narrow<size_t>(allocationSize));
#endif // DML_USE_HEAP_ALLOCATOR_FOR_UNPOOLED_MEMORY
        }

        // Get resource ID
        uint64_t resourceId;
        auto resourcePointer = uintptr_t(resourceWrapper->GetD3D12Resource());
        auto [it, isNew] = m_resourceIds.emplace(resourcePointer, m_currentResourceId + 1);
        if (isNew)
        {
            resourceId = ++m_currentResourceId;
        }
        else
        {
            resourceId = it->second;
        }

        // Create allocation info
        ComPtr<AllocationInfo> allocInfo =
            wil::MakeOrThrow<AllocationInfo>(this, ++m_currentAllocationId, resourceId, resourceWrapper.Get(), size);

        return allocInfo.Detach();
    }

    void TiledBufferAllocator::FreeResource(void* p, uint64_t resourceId)
    {
        AllocationInfo* allocInfo = static_cast<AllocationInfo*>(p);

        assert(allocInfo != nullptr); // Can't free nullptr

        if (allocInfo->GetOwner() != this)
        {
            // This allocation doesn't belong to this allocator!
            ORT_THROW_HR(E_INVALIDARG);
        }

        if (!m_pooledAllocator.TryReleaseBuffer(allocInfo->GetResource()))
        {
#ifdef DML_USE_HEAP_ALLOCATOR_FOR_UNPOOLED_MEMORY
            m_unpooledAllocator.TryReleaseBuffer(allocInfo->GetResource());
#else
            // Free the underlying allocation once queued work has completed.
    #ifdef _GAMING_XBOX
            m_context->QueueReference(WRAP_GRAPHICS_UNKNOWN(allocInfo->GetResource()).Get());
    #else
            m_context->QueueReference(allocInfo->GetResource());
    #endif
#endif

            m_resourceIds.erase(uintptr_t(allocInfo->GetResource()));
        }

        allocInfo->DetachResourceWrapper();
    }

    void TiledBufferAllocator::FreeResources(std::vector<Microsoft::WRL::ComPtr<IUnknown>>& resources)
    {
        for (auto& resource : resources)
        {
            m_resourceIds.erase(uintptr_t(resource.Get()));
            m_context->QueueReference(resource.Get());
        }
    }
} // namespace Dml