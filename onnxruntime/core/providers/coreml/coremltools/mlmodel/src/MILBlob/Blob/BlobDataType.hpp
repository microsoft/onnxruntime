// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "MILBlob/Bf16.hpp"
#include "MILBlob/Fp16.hpp"

namespace MILBlob {
namespace Blob {

enum class BlobDataType : uint32_t
{
    // *** WARNING ***
    // for binary compatibility, values should ONLY be added at the end.
    //
    // this file needs to remain in sync across multiple repos.
    // please be cognizant of that when making changes to the
    // format.

    Float16 = 1,
    Float32 = 2,
    UInt8 = 3,
    Int8 = 4,
    BFloat16 = 5,
    Int16 = 6,
    UInt16 = 7,
};

template <typename T>
struct BlobDataTypeTraits;

template <>
struct BlobDataTypeTraits<float> {
    static constexpr BlobDataType DataType = BlobDataType::Float32;
};

template <>
struct BlobDataTypeTraits<Fp16> {
    static constexpr BlobDataType DataType = BlobDataType::Float16;
};

template <>
struct BlobDataTypeTraits<Bf16> {
    static constexpr BlobDataType DataType = BlobDataType::BFloat16;
};

template <>
struct BlobDataTypeTraits<uint8_t> {
    static constexpr BlobDataType DataType = BlobDataType::UInt8;
};

template <>
struct BlobDataTypeTraits<int8_t> {
    static constexpr BlobDataType DataType = BlobDataType::Int8;
};

template <>
struct BlobDataTypeTraits<int16_t> {
    static constexpr BlobDataType DataType = BlobDataType::Int16;
};

template <>
struct BlobDataTypeTraits<uint16_t> {
    static constexpr BlobDataType DataType = BlobDataType::UInt16;
};

}  // namespace Blob
}  // namespace MILBlob
