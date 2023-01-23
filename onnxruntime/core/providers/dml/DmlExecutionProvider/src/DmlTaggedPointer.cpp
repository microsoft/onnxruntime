// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlTaggedPointer.h"
#include <cassert>

namespace Dml
{
/*static*/ TaggedPointer TaggedPointer::Unpack(const void* ptr)
{
    uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);

    static constexpr uint64_t kAllocationIDMask =
        (1ull << kAllocationIDBits) - 1;
    static constexpr uint64_t kOffsetMask = (1ull << kOffsetBits) - 1;

    TaggedPointer tagged_ptr;
    tagged_ptr.allocation_id = (ptr_val >> kOffsetBits) & kAllocationIDMask;
    tagged_ptr.offset = (ptr_val & kOffsetMask);

    return tagged_ptr;
}

/*static*/ void* TaggedPointer::Pack(uint32_t allocation_id, uint64_t offset)
{
    assert(allocation_id < (1ull << kAllocationIDBits));
    assert(offset < (1ull << kOffsetBits));
    uint64_t ptr = ((uint64_t)allocation_id << kOffsetBits) | offset;

    return reinterpret_cast<void*>(ptr);
}
}
