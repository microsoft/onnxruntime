#include "precomp.h"

#include "core/session/onnxruntime_c_api.h"

#include "TiledMemoryAllocator.h"
#include "DmlSubAllocator.h"

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
      onnxruntime::IAllocator(
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
  { }

  const AllocationInfo * TiledBufferAllocator::DecodeDataHandle(const void * opaqueHandle)
  {
    if (opaqueHandle == nullptr)
    {
      // There is no memory allocated which needs to be decoded.
      ORT_THROW_HR(E_INVALIDARG);
    }
    const auto* allocInfo = static_cast<const AllocationInfo*>(opaqueHandle);
    return allocInfo;
  }

  void TiledBufferAllocator::SetDefaultRoundingMode(AllocatorRoundingMode roundingMode)
  {
    m_defaultRoundingMode = roundingMode;
  }

  void TiledBufferAllocator::SetResidency(bool value)
  {
  }

  void* TiledBufferAllocator::Alloc(size_t size)
  {
    return Alloc(size, m_defaultRoundingMode);
  }

  void * TiledBufferAllocator::Alloc(size_t size, AllocatorRoundingMode roundingMode)
  {
    return nullptr;
  }
}