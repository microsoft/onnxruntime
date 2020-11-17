// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "robuffer.h"
#include "winrt/Windows.Storage.Streams.h"

namespace _winml {

class data_buffer_container {
 public:
  static std::shared_ptr<data_buffer_container> create(
    size_t num_elements,
    size_t element_size_in_bytes,
    wfc::IIterable<wss::IBuffer> const& buffers);

private:
  // Privte constructor as this type should be created as a shared_ptr
  data_buffer_container(size_t num_elements, size_t element_size_in_bytes, wfc::IIterable<wss::IBuffer> const& buffers);
  gsl::span<byte> buffer_at(size_t index);
  gsl::span<byte> combined_buffer();

 public:
  size_t num_elements();
  size_t size_in_bytes();
  size_t num_buffers();

  // Buffer accessors
  std::vector<wss::IBuffer>& buffers();
  gsl::span<byte> buffer(bool should_sync_buffer);

  // Flush to buffers API
  bool flush();

  // Set APIs
  void set(size_t size_in_bytes, const T* data);

  template <typename T>
  auto set(std::vector<T>&& moveableData) {
    set(moveableData.size() * element_size_in_bytes_, moveableData.data());
  }

 private:
  wss::IBuffer combined_buffer_;
  std::vector<wss::IBuffer> buffers_;
  size_t num_elements_;
  size_t element_size_in_bytes_;
};

}  // namespace _winml