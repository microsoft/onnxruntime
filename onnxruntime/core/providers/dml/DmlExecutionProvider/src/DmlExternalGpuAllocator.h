// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace Dml
{
    class DmlReservedResourceSubAllocator;
    class AllocationInfo;
    struct TaggedPointer;

    class DmlExternalGpuAllocator : public onnxruntime::IAllocator
    {
    public:
        DmlExternalGpuAllocator();

        void* Alloc(size_t size_in_bytes) final;
        void Free(void* ptr) final;
    };
} // namespace Dml
