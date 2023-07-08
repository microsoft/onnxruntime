// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlAllocationInfo.h"
#include "BucketizedBufferAllocator.h"

namespace Dml
{

    AllocationInfo::~AllocationInfo()
    {
        if (m_owner)
        {
            m_owner->FreeResource(this);
        }
    }

} // namespace Dml
