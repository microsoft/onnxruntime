// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "DmlTaggedPointer.h"
#include <cassert>

namespace Dml
{
/*static*/ TaggedPointer TaggedPointer::Unpack(const void* ptr)
{
    uint64_t ptrVal = reinterpret_cast<uint64_t>(ptr);

    static constexpr uint64_t allocationIDMask = (1ull << AllocationIDBits) - 1;
    static constexpr uint64_t offsetMask = (1ull << OffsetBits) - 1;

    TaggedPointer taggedPtr;
    taggedPtr.deviceId = (ptrVal >> (AllocationIDBits + OffsetBits));
    taggedPtr.allocationId = (ptrVal >> OffsetBits) & allocationIDMask;
    taggedPtr.offset = (ptrVal & offsetMask);

    return taggedPtr;
}

/*static*/ void* TaggedPointer::Pack(uint32_t deviceId, uint32_t allocationId, uint64_t offset)
{
    assert(deviceId < (1ull << DeviceIDBits));
    assert(allocationId < (1ull << AllocationIDBits));
    assert(offset < (1ull << OffsetBits));

    // Store the device ID in the upper bits of the pointer, followed by the
    // allocation id and the offset in the lower bits
    uint64_t ptr = ((uint64_t)deviceId << (AllocationIDBits + OffsetBits)) |
                   ((uint64_t)allocationId << OffsetBits) | offset;

    return reinterpret_cast<void*>(ptr);
}

uint64_t TaggedPointer::GetUniqueId() const
{
    return reinterpret_cast<uint64_t>(TaggedPointer::Pack(deviceId, allocationId, offset));
}

} // namespace tfdml
