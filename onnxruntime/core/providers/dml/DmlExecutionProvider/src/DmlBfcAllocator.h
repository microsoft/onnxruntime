// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlHeapAllocator.h"

namespace Dml
{
    // A simple wrapper that will be wrapped by ORT to create a BFC allocator
    class DmlBfcAllocator : public onnxruntime::IAllocator
    {
    public:
        DmlBfcAllocator(D3D12HeapAllocator* heapAllocator)
            : onnxruntime::IAllocator(
                OrtMemoryInfo(
                    "DML",
                    OrtAllocatorType::OrtDeviceAllocator,
                    OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)
                )
            ),
            m_heapAllocator(heapAllocator) {}

        void* Alloc(size_t sizeInbytes) final { return m_heapAllocator->Alloc(sizeInbytes); }
        void Free(void* ptr) final { m_heapAllocator->Free(ptr); };

    private:
        D3D12HeapAllocator* m_heapAllocator;
    };
}
