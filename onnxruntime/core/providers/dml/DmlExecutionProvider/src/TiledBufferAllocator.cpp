#include "precomp.h"

#include "core/session/onnxruntime_c_api.h"

#include "TiledBufferAllocator.h"
#include "DmlSubAllocator.h"
#include "DmlCommittedResourceWrapper.h"

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
      m_heapAllocator(device, context->Queue())
  { }

  void TiledBufferAllocator::SetResidency(bool value)
  {
      m_subAllocator->SetResidency(value);
      m_heapAllocator.SetResidency(value);
  }

  void * TiledBufferAllocator::Alloc(size_t size, AllocatorRoundingMode roundingMode)
  {
    // For some reason lotus likes requesting 0 bytes of memory
    size = std::max<size_t>(1, size);

    // Use a pooled resource if the size (post rounding, if requested) matches a bucket size
    ComPtr<DmlResourceWrapper> resourceWrapper;
    if (roundingMode == AllocatorRoundingMode::Enabled)
    {
      wil::MakeOrThrow<DmlCommittedResourceWrapper>(m_heapAllocator.AllocateBuffer(size)).As(&resourceWrapper);      
    }
    else
    {
      // The allocation will not be pooled.  Construct a new one
      auto allocationSize = (size + 3) & ~3;
      resourceWrapper = m_subAllocator->Alloc(onnxruntime::narrow<size_t>(allocationSize));
    }

    ComPtr<AllocationInfo> allocInfo = wil::MakeOrThrow<AllocationInfo>(
      this,
      ++m_currentAllocationId,
      m_resourceIds.emplace(uintptr_t(resourceWrapper->GetD3D12Resource()), m_resourceIds.size()).first->second,
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
      // Free the underlying allocation once queued work has completed.
#ifdef _GAMING_XBOX
      m_context->QueueReference(WRAP_GRAPHICS_UNKNOWN(allocInfo->GetResource()).Get());
#else
      m_context->QueueReference(allocInfo->GetResource());
#endif
      allocInfo->DetachResourceWrapper();
    }
  }
}