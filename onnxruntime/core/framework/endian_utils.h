// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstring>
#include <type_traits>

#include "gsl/gsl"

#include "core/common/status.h"
#include "core/framework/endian.h"

namespace onnxruntime {
namespace utils {

/**
 * Swaps the byte order of elements in a buffer.
 * This is a low-level funtion - please be sure to pass in valid arguments.
 * In particular, source_bytes and destination_bytes should have the same size,
 * which should be a multiple of element_size_in_bytes. element_size_in_bytes
 * should also be greater than zero.
 *
 * @param element_size_in_bytes The size of an individual element, in bytes.
 * @param source_bytes The source byte span.
 * @param destination_bytes The destination byte span.
 */
void SwapByteOrderCopy(size_t element_size_in_bytes, gsl::span<const char> source_bytes, gsl::span<char> destination_bytes);

/**
 * Copies between two buffers where one is little-endian and the other has
 * native endian-ness.
 */
template <size_t ElementSize>
common::Status CopyLittleEndian(gsl::span<const char> source_bytes, gsl::span<char> destination_bytes) {
  if (endian::native == endian::little) {
    std::memcpy(destination_bytes.data(), source_bytes.data(), source_bytes.size_bytes());
  } else {
    SwapByteOrderCopy(ElementSize, source_bytes, destination_bytes);
  }
  return Status::OK();
}

/**
 * Reads from a little-endian source.
 */
template <typename T>
common::Status ReadLittleEndian(gsl::span<const char> source_bytes, gsl::span<T> destination) {
  static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
  const auto destination_bytes = gsl::make_span(
      reinterpret_cast<char*>(destination.data()), destination.size_bytes());
  return CopyLittleEndian<sizeof(T)>(source_bytes, destination_bytes);
}

/**
 * Writes to a little-endian destination.
 */
template <typename T>
common::Status WriteLittleEndian(gsl::span<const T> source, gsl::span<char> destination_bytes) {
  static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
  const auto source_bytes = gsl::make_span(
      reinterpret_cast<const char*>(source.data()), source.size_bytes());
  return CopyLittleEndian<sizeof(T)>(source_bytes, destination_bytes);
}

}  // namespace utils
}  // namespace onnxruntime
