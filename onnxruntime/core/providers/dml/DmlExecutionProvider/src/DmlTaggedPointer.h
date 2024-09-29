// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <climits>
#include <cstdint>

namespace Dml
{

// D3D12HeapAllocator and D3D12DescriptorHeapAllocator encode the allocation ID
// into the high bits of the pointers it returns, while the low bits are used as
// an offset into the allocation. Note that since the layout of bitfields is
// implementation-defined, you can't just cast a void* into a TaggedPointer: it
// must be done using masks and shifts.
struct TaggedPointer
{
    static constexpr uint64_t DeviceIDBits = 4;
    static constexpr uint64_t AllocationIDBits = 20;
    static constexpr uint64_t OffsetBits = 40;

    uint64_t deviceId : DeviceIDBits;
    uint64_t allocationId : AllocationIDBits;
    uint64_t offset : OffsetBits;

    static void* Pack(uint32_t deviceId, uint32_t allocationId, uint64_t offset);
    static TaggedPointer Unpack(const void* ptr);
    uint64_t GetUniqueId() const;
};

static_assert(
    sizeof(TaggedPointer) == sizeof(void*),
    "DML requires a 64-bit architecture");
static_assert(
    TaggedPointer::DeviceIDBits + TaggedPointer::AllocationIDBits + TaggedPointer::OffsetBits == sizeof(void*) * CHAR_BIT,
    "DML requires a 64-bit architecture");

} // namespace tfdml
