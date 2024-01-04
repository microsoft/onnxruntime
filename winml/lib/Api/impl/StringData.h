// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "IData.h"
#include "robuffer.h"
#include "winrt/Windows.Storage.Streams.h"

namespace _winml {

class string_data : public _winml::idata {
 public:
  static std::shared_ptr<_winml::idata> create(size_t size);

  string_data(size_t size);

  size_t num_elements() override;
  size_t size_in_bytes() override;
  size_t num_buffers() override;

  // Buffer accessors
  std::vector<wss::IBuffer>& buffers() override;
  gsl::span<byte> buffer(bool should_sync_buffer) override;

  // Flush to buffers API
  bool flush() override;

  // Set APIs
  void set(size_t data_size, const byte* data) override;

 public:
  void set(size_t num_elements, const std::string_view* data);
  std::vector<std::string>& get_backing_vector();

 private:
  std::vector<std::string> buffer_;
};

}  // namespace _winml
