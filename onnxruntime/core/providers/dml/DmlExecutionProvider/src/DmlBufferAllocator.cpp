#include "precomp.h"

#include "DmlBufferAllocator.h"

namespace Dml
{
    AllocationInfo::~AllocationInfo()
    {
        if (m_owner)
        {
            m_owner->FreeResource(this, m_pooledResourceId);
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

    void DmlBufferAllocator::SetDefaultRoundingMode(AllocatorPoolingMode poolingMode)
    {
        m_defaultPoolingMode = poolingMode;
    }

    const AllocationInfo* DmlBufferAllocator::DecodeDataHandle(const void* opaqueHandle)
    {
        if (opaqueHandle == nullptr)
        {
            // There is no memory allocated which needs to be decoded.
            ORT_THROW_HR(E_INVALIDARG);
        }
        const auto* allocInfo = static_cast<const AllocationInfo*>(opaqueHandle);
        return allocInfo;
    }

    void* DmlBufferAllocator::Alloc(size_t size)
    {
        return Alloc(size, m_defaultPoolingMode);
    }

    void DmlBufferAllocator::Free(void* p)
    {
        // Release Lotus's reference on the allocation.  The allocation
        // also inherits IUnknown, and once its final reference reaches zero
        // it will call FreeResource
        ComPtr<AllocationInfo> allocInfo;
        allocInfo.Attach(static_cast<AllocationInfo*>(p));
    }
}