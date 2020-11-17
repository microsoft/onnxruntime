// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

#include "StringBufferContainer.h"

namespace _winml {

string_buffer_container::string_buffer_container(size_t size) :
    buffer_(size) {}

std::shared_ptr<string_buffer_container> string_buffer_container::create(size_t size) {
  return std::make_shared<string_buffer_container>(size);
}

size_t string_buffer_container::num_elements() {
  return buffer_.size();
}

size_t string_buffer_container::size_in_bytes() {
  WINML_THROW_HR(E_UNEXPECTED);
}

size_t string_buffer_container::num_buffers() {
  return 1;
}

bool string_buffer_container::flush() {
  // Vacuously true
  return true;
}

gsl::span<byte> string_buffer_container::buffer(bool /*should_sync_buffer*/) {
  return gsl::span<byte>(reinterpret_cast<byte*>(buffer_.data()), buffer_.size());
}

void string_buffer_container::set(size_t size, std::string_view* data) {
  WINML_THROW_HR_IF_FALSE_MSG(
      E_INVALIDARG,
      size <= buffer_.size(),
      "Argument size (%d) exceeds the tensor size (%d).",
      static_cast<int>(size),
      static_cast<int>(buffer_.size()));

  // Copy
  std::copy(data, data + size, buffer_.begin());
}

}  // namespace _winml