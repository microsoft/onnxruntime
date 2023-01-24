// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "BucketizedBufferAllocator.h"

namespace Dml
{
    class DmlBfcAllocator : public onnxruntime::IAllocator
    {
    public:
        DmlBfcAllocator(BucketizedBufferAllocator* subAllocator)
        : onnxruntime::IAllocator(
            OrtMemoryInfo(
                "DML",
                OrtAllocatorType::OrtDeviceAllocator,
                OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)
            )
        ),
        m_subAllocator(subAllocator) {}

        void* Alloc(size_t size_in_bytes) { return m_subAllocator->Alloc(size_in_bytes); }
        void Free(void* ptr) { m_subAllocator->Free(ptr); }
    private:
        BucketizedBufferAllocator* m_subAllocator;
    };
} // namespace Dml
