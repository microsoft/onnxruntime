// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <cstdint>
#include <functional>

namespace MILBlob {

/**
 * Struct for holding bytes that represent a fp16 number.
 * Floating point interface treats "bytes" as IEEE 754 half precision floating point
 *  (https://ieeexplore.ieee.org/document/8766229)
 */
struct Fp16 {
    explicit Fp16(uint16_t bs) : bytes(bs) {}
    Fp16() : bytes(0) {}

    static Fp16 FromFloat(float f);

    float GetFloat() const;
    void SetFloat(float f);

    // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
    uint16_t bytes;
};

inline bool operator==(const Fp16& first, const Fp16& second) noexcept
{
    return first.bytes == second.bytes;
}

inline bool operator!=(const Fp16& first, const Fp16& second) noexcept
{
    return first.bytes != second.bytes;
}

}  // namespace MILBlob

namespace std {

template <>
struct hash<MILBlob::Fp16> {
    size_t operator()(const MILBlob::Fp16& fp) const
    {
        return fp.bytes;
    }
};

}  // namespace std
