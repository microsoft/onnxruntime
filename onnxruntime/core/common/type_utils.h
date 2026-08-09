// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <type_traits>

namespace onnxruntime {
namespace utils {
template <typename T>
struct IsByteType : std::false_type {};

template <>
struct IsByteType<uint8_t> : std::true_type {};

template <>
struct IsByteType<int8_t> : std::true_type {};

}  // namespace utils
}  // namespace onnxruntime
