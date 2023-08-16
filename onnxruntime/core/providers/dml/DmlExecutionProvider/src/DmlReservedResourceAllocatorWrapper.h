// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "DmlReservedResourceSubAllocator.h"

namespace Dml
{
    class DmlReservedResourceAllocatorWrapper : public onnxruntime::IAllocator
    {
    public:
        DmlReservedResourceAllocatorWrapper(std::shared_ptr<DmlReservedResourceSubAllocator> subAllocator)
        : onnxruntime::IAllocator(
            OrtMemoryInfo(
                onnxruntime::DML,
                OrtAllocatorType::OrtDeviceAllocator,
                OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0)
            )
        ),
        m_subAllocator(std::move(subAllocator)) {}

        void* Alloc(size_t sizeInBytes) final { return m_subAllocator->Alloc(sizeInBytes); }
        void Free(void* ptr) final { m_subAllocator->Free(ptr); }
    private:
        std::shared_ptr<DmlReservedResourceSubAllocator> m_subAllocator;
    };
} // namespace Dml
