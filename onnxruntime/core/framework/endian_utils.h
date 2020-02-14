// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <type_traits>

#include "gsl/gsl"

#include "core/common/status.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace utils {

/**
 * Copies elements and swaps their byte orders.
 *
 * This is a low-level function - please be sure to pass in valid arguments.
 * In particular:
 * - source_bytes and destination_bytes should have the same size, which should
 *   be a multiple of element_size_in_bytes.
 * - element_size_in_bytes should be greater than zero.
 * - source_bytes and destination_bytes should not overlap.
 *
 * @param element_size_in_bytes The size of an individual element, in bytes.
 * @param source_bytes The source byte span.
 * @param destination_bytes The destination byte span.
 */
void SwapByteOrderCopy(
    size_t element_size_in_bytes, gsl::span<const char> source_bytes, gsl::span<char> destination_bytes);

namespace detail {

/**
 * Copies between two buffers where one is little-endian and the other has
 * native endian-ness.
 */
void CopyLittleEndian(
    size_t element_size_in_bytes, gsl::span<const char> source_bytes, gsl::span<char> destination_bytes);

}  // namespace detail

/**
 * Reads from a little-endian source.
 */
template <typename T>
common::Status ReadLittleEndian(gsl::span<const char> source_bytes, gsl::span<T> destination) {
// std::is_trivially_copyable is not implemented in older versions of GCC
#if !defined(__GNUC__) || __GNUC__ >= 5
  static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
#endif
  ORT_RETURN_IF_NOT(source_bytes.size_bytes() == destination.size_bytes(),
                    "source and destination buffer size mismatch");
  const auto destination_bytes = gsl::make_span(
      reinterpret_cast<char*>(destination.data()), destination.size_bytes());
  detail::CopyLittleEndian(sizeof(T), source_bytes, destination_bytes);
  return common::Status::OK();
}

/**
 * Writes to a little-endian destination.
 */
template <typename T>
common::Status WriteLittleEndian(gsl::span<const T> source, gsl::span<char> destination_bytes) {
// std::is_trivially_copyable is not implemented in older versions of GCC
#if !defined(__GNUC__) || __GNUC__ >= 5
  static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
#endif
  ORT_RETURN_IF_NOT(source.size_bytes() == destination_bytes.size_bytes(),
                    "source and destination buffer size mismatch");
  const auto source_bytes = gsl::make_span(
      reinterpret_cast<const char*>(source.data()), source.size_bytes());
  detail::CopyLittleEndian(sizeof(T), source_bytes, destination_bytes);
  return common::Status::OK();
}

}  // namespace utils
}  // namespace onnxruntime
