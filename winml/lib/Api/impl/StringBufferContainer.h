// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "robuffer.h"
#include "winrt/Windows.Storage.Streams.h"

namespace _winml {

class string_buffer_container {
 public:
  static auto create(size_t size);

 private:
  string_buffer_container(size_t size) :
    buffer_(size) {}

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

 private:
  std::vector<std::string> buffer_;
};

}  // namespace _winml