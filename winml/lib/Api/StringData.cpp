// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api/pch/pch.h"

#include "impl/StringData.h"

namespace _winml {

string_data::string_data(size_t size) : buffer_(size) {
}

std::shared_ptr<_winml::idata> string_data::create(size_t size) {
  return std::make_shared<string_data>(size);
}

size_t string_data::num_elements() {
  return buffer_.size();
}

size_t string_data::size_in_bytes() {
  WINML_THROW_HR(E_UNEXPECTED);
}

size_t string_data::num_buffers() {
  return 1;
}

bool string_data::flush() {
  // Vacuously true
  return true;
}

std::vector<wss::IBuffer>& string_data::buffers() {
  WINML_THROW_HR(E_UNEXPECTED);
}

gsl::span<byte> string_data::buffer(bool /*should_sync_buffer*/) {
  return gsl::span<byte>(reinterpret_cast<byte*>(buffer_.data()), buffer_.size());
}

void string_data::set(size_t num_elements, const std::string_view* data) {
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    num_elements <= buffer_.size(),
    "Argument size (%d) exceeds the tensor size (%d).",
    static_cast<int>(num_elements),
    static_cast<int>(buffer_.size())
  );

  // Copy
  std::copy(data, data + num_elements, buffer_.begin());
}

void string_data::set(size_t /*data_size*/, const byte* /*data*/) {
  WINML_THROW_HR(E_UNEXPECTED);
}

std::vector<std::string>& string_data::get_backing_vector() {
  return buffer_;
}

}  // namespace _winml
