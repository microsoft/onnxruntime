#include "precomp.h"

#include "core/session/onnxruntime_c_api.h"

#include "TiledBufferAllocator.h"
#include "DmlSubAllocator.h"
#include "DmlCommittedResourceWrapper.h"

#include <format>
#define DebugOut(pattern, ...) OutputDebugString(std::format(pattern L"\n", __VA_ARGS__).c_str())

namespace Dml
{
  TiledBufferAllocator::TiledBufferAllocator(
    ID3D12Device * device,
    std::shared_ptr<ExecutionContext> context,
    const D3D12_HEAP_PROPERTIES & heapProps,
    D3D12_HEAP_FLAGS heapFlags,
    D3D12_RESOURCE_FLAGS resourceFlags,
    D3D12_RESOURCE_STATES initialState,
    std::unique_ptr<DmlSubAllocator>&& subAllocator) :
      DmlBufferAllocator(
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
      m_subAllocator(std::move(subAllocator)),
      m_heapAllocator(device, context->Queue()),
      m_longAllocator(device, context->Queue())
  { }

  void TiledBufferAllocator::SetResidency(bool value)
  {
      m_subAllocator->SetResidency(value);
      m_heapAllocator.SetResidency(value);
      m_longAllocator.SetResidency(value);

      if (!value)
      {
          auto resources = m_heapAllocator.Clear();
          for (auto& resource : resources)
          {
              m_resourceIds.erase(uintptr_t(resource.Get()));
              m_context->QueueReference(resource.Get());
          }

          m_tiledAllocationSize = 0;
      }
  }

  void * TiledBufferAllocator::Alloc(size_t size, AllocatorRoundingMode roundingMode)
  {
    // For some reason lotus likes requesting 0 bytes of memory
    size = std::max<size_t>(1, size);

    bool isSizePooled = size >= 65536 && !(size & (size - 1));

    // Use a pooled resource if the size (post rounding, if requested) matches a bucket size
    ComPtr<DmlResourceWrapper> resourceWrapper;
    if (roundingMode == AllocatorRoundingMode::Enabled)
    {
      m_tiledAllocationSize += size;
      DebugOut(L"!!! Tiled allocate {:.2f} / {:.2f}", size / 1024.f / 1024.f, m_tiledAllocationSize / 1024.f / 1024.f);
      wil::MakeOrThrow<DmlCommittedResourceWrapper>(m_heapAllocator.AllocateBuffer(size)).As(&resourceWrapper);      
    }
    else
    {
      m_untiledAllocationSize += size;
      DebugOut(L"!!! Untiled allocate {:.2f} / {:.2f}", size / 1024.f / 1024.f, m_untiledAllocationSize / 1024.f / 1024.f);

      // The allocation will not be pooled.  Construct a new one
      /*auto allocationSize = (size + 3) & ~3;
      resourceWrapper = m_subAllocator->Alloc(onnxruntime::narrow<size_t>(allocationSize));*/

      wil::MakeOrThrow<DmlCommittedResourceWrapper>(m_longAllocator.AllocateBuffer(size)).As(&resourceWrapper);
    }

    //Get resource ID
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

    //Create allocation info
    ComPtr<AllocationInfo> allocInfo = wil::MakeOrThrow<AllocationInfo>(
      this,
      ++m_currentAllocationId,
      resourceId,
      resourceWrapper.Get(),
      size
    );

    return allocInfo.Detach();
  }

  void TiledBufferAllocator::FreeResource(void * p, uint64_t resourceId)
  {
    AllocationInfo *allocInfo = static_cast<AllocationInfo*>(p);
    
    assert(allocInfo != nullptr); // Can't free nullptr

    if (allocInfo->GetOwner() != this)
    {
      // This allocation doesn't belong to this allocator!
      ORT_THROW_HR(E_INVALIDARG);
    }

    if (!m_heapAllocator.TryReleaseBuffer(allocInfo->GetResource()))
    {
      m_untiledAllocationSize -= allocInfo->GetRequestedSize();
      DebugOut(L"!!! Untiled deallocate {:.2f} / {:.2f}", allocInfo->GetRequestedSize() / 1024.f / 1024.f, m_untiledAllocationSize / 1024.f / 1024.f);
      // Free the underlying allocation once queued work has completed.
//#ifdef _GAMING_XBOX
//      m_context->QueueReference(WRAP_GRAPHICS_UNKNOWN(allocInfo->GetResource()).Get());
//#else
//      m_context->QueueReference(allocInfo->GetResource());
//#endif

      m_longAllocator.TryReleaseBuffer(allocInfo->GetResource());

      m_resourceIds.erase(uintptr_t(allocInfo->GetResource()));
    }
    else
    {
      m_tiledAllocationSize -= allocInfo->GetRequestedSize();
      DebugOut(L"!!! Tiled deallocate {:.2f} / {:.2f}", allocInfo->GetRequestedSize() / 1024.f / 1024.f, m_tiledAllocationSize / 1024.f / 1024.f);
    }
    
    allocInfo->DetachResourceWrapper();
  }
}