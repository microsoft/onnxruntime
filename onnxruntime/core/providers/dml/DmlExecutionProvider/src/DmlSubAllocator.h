// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Dml
{
    struct DmlResourceWrapper;

    class DmlSubAllocator
    {
    public:
        virtual void FreeResource(AllocationInfo* allocInfo, uint64_t resourceId) = 0;
    };
}
