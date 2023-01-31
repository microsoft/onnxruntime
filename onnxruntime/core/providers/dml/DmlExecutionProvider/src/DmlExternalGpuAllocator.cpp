// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "precomp.h"
#include "DmlExternalGpuAllocator.h"

namespace Dml
{
    DmlExternalGpuAllocator::DmlExternalGpuAllocator()
    : onnxruntime::IAllocator(
        OrtMemoryInfo(
            onnxruntime::DML,
            OrtAllocatorType::OrtDeviceAllocator,
            OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DML_EXTERNAL, 0),
            -1
        )
    ) {}

    void* DmlExternalGpuAllocator::Alloc(size_t size_in_bytes)
    {
        // This allocator should never be used to allocate memory; it should only be use to decode the opaque data pointer
        THROW_HR(E_INVALIDARG);
    }

    void DmlExternalGpuAllocator::Free(void* ptr)
    {
        // This allocator should never be used to free memory; it should only be use to decode the opaque data pointer
        THROW_HR(E_INVALIDARG);
    }

} // namespace Dml
