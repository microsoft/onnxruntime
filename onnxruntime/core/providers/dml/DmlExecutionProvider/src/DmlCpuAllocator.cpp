// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlCpuAllocator.h"

namespace Dml
{

DmlCpuAllocator::DmlCpuAllocator(OrtMemType memType)
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

void* DmlCpuAllocator::Alloc(size_t size)
{
    if (size <= 0)
    {
        return nullptr;
    }
    void* p = malloc(size);
    return p;
}

void DmlCpuAllocator::Free(void* p)
{
    free(p);
}

} // namespace Dml
