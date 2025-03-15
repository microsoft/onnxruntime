// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/endian_utils.h"

#include <cassert>
#include <cstring>

#include "core/framework/endian.h"

namespace onnxruntime {
namespace utils {

namespace {

// analogous to std::reverse_copy
template <typename BidirIt, typename OutputIt>
OutputIt ReverseCopy(BidirIt first, BidirIt last, OutputIt d_first) {
  while (last != first) {
    --last;
    *d_first = *last;
    ++d_first;
  }
  return d_first;
}

}  // namespace

void SwapByteOrderCopy(size_t element_size_in_bytes,
                       gsl::span<const unsigned char> source_bytes,
                       gsl::span<unsigned char> destination_bytes) {
  assert(element_size_in_bytes > 0);
  assert(source_bytes.size_bytes() % element_size_in_bytes == 0);
  assert(source_bytes.size_bytes() == destination_bytes.size_bytes());
  // check non-overlapping
  // given begin <= end, end0 <= begin1 || end1 <= begin0
  assert(source_bytes.data() + source_bytes.size() <= destination_bytes.data() ||
         destination_bytes.data() + destination_bytes.size() <= source_bytes.data());

  for (size_t element_offset = 0, element_offset_end = source_bytes.size_bytes();
       element_offset < element_offset_end;
       element_offset += element_size_in_bytes) {
    const auto source_element_bytes = source_bytes.subspan(element_offset, element_size_in_bytes);
    const auto dest_element_bytes = destination_bytes.subspan(element_offset, element_size_in_bytes);
    ReverseCopy(source_element_bytes.data(),
                source_element_bytes.data() + source_element_bytes.size_bytes(),
                dest_element_bytes.data());
  }
}

void SwapByteOrderInplace(size_t element_size_in_bytes, gsl::span<unsigned char> bytes) {
  assert(bytes.size_bytes() % element_size_in_bytes == 0);
  auto* data = bytes.data();
  for (size_t i = 0, num_elements = bytes.size(); i < num_elements; ++i) {
    auto* start_byte = data + i * element_size_in_bytes;
    auto* end_byte = start_byte + element_size_in_bytes - 1;
    for (size_t count = 0; count < element_size_in_bytes / 2; ++count) {
      std::swap(*start_byte++, *end_byte--);
    }
  }
}

namespace detail {

Status CopyLittleEndian(size_t element_size_in_bytes,
                        gsl::span<const unsigned char> source_bytes,
                        gsl::span<unsigned char> destination_bytes) {
  ORT_RETURN_IF(source_bytes.size_bytes() != destination_bytes.size_bytes(),
                "source and destination buffer size mismatch");

  if constexpr (endian::native == endian::little) {
    std::memcpy(destination_bytes.data(), source_bytes.data(), source_bytes.size_bytes());
  } else {
    SwapByteOrderCopy(element_size_in_bytes, source_bytes, destination_bytes);
  }

  return Status::OK();
}

}  // namespace detail

common::Status ReadLittleEndian(size_t element_size,
                                gsl::span<const unsigned char> source_bytes,
                                gsl::span<unsigned char> destination_bytes) {
  return detail::CopyLittleEndian(element_size, source_bytes, destination_bytes);
}

}  // namespace utils
}  // namespace onnxruntime
