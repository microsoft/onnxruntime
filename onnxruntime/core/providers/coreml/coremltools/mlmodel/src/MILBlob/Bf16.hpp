// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <cstdint>
#include <functional>

namespace MILBlob {

/**
 * Struct for holding bytes that represent a bf16 number.
 * Floating point interface treats "bytes" as brain float16 floating point
 *  (https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
 */
struct Bf16 {
    explicit Bf16(uint16_t bs) : bytes(bs) {}
    Bf16() : bytes(0) {}

    static Bf16 FromFloat(float f);

    float GetFloat() const;
    void SetFloat(float f);

    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
    uint16_t bytes;
};

inline bool operator==(const Bf16& first, const Bf16& second) noexcept
{
    // Note this comparison is quick and dirty - it will give incorrect results
    // for (-0.0 == 0.0) and, depending on bit pattern, (NaN == NaN).
    return first.bytes == second.bytes;
}

inline bool operator!=(const Bf16& first, const Bf16& second) noexcept
{
    // Note this comparison is quick and dirty - it will give incorrect results
    // for (-0.0 != 0.0) and, depending on bit pattern, (NaN != NaN).
    return first.bytes != second.bytes;
}

}  // namespace MILBlob

namespace std {

template <>
struct hash<MILBlob::Bf16> {
    size_t operator()(const MILBlob::Bf16& fp) const
    {
        return fp.bytes;
    }
};

}  // namespace std
