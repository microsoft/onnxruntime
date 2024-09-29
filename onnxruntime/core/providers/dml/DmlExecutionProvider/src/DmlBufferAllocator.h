#pragma once
#include "core/framework/allocator.h"
#include "AllocationInfo.h"

namespace Dml
{
    class CPUAllocator : public onnxruntime::IAllocator
    {
    public:
        explicit CPUAllocator(OrtMemType memType);

        void* Alloc(size_t size) override;
        void Free(void* p) override;
    };

    class DmlBufferAllocator : public onnxruntime::IAllocator
    {
    public:
        void SetDefaultRoundingMode(AllocatorPoolingMode poolingMode);

        // Returns the information associated with an opaque allocation handle returned by IAllocator::Alloc.
        const AllocationInfo* DecodeDataHandle(const void* opaqueHandle);

        void* Alloc(size_t size) final;
        virtual void* Alloc(size_t size, AllocatorPoolingMode poolingMode) = 0;
        void Free(void* p) final;

        virtual DmlAllocatorType Type() const = 0;

        virtual void Clean() { };

    protected:
        using onnxruntime::IAllocator::IAllocator;

        // Unless specifically requested, allocation sizes are not rounded to enable pooling
        // until SetDefaultRoundingMode is called.  This should be done at completion of session
        // initialization.
        AllocatorPoolingMode m_defaultPoolingMode = AllocatorPoolingMode::Disabled;

        friend class AllocationInfo;
        virtual void FreeResource(void* p, uint64_t resourceId) = 0;
    };
}