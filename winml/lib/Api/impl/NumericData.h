// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "IData.h"
#include "robuffer.h"
#include "winrt/Windows.Storage.Streams.h"

namespace _winml {

class numeric_data : public _winml::idata {
 public:
  static std::shared_ptr<_winml::idata> create(
    size_t num_elements, size_t element_size_in_bytes, wfc::IIterable<wss::IBuffer> const& buffers
  );

  // Privte constructor as this type should be created as a shared_ptr
  numeric_data(size_t num_elements, size_t element_size_in_bytes, wfc::IIterable<wss::IBuffer> const& buffers);
  gsl::span<byte> buffer_at(size_t index);
  gsl::span<byte> combined_buffer();

 public:
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

 private:
  wss::IBuffer combined_buffer_;
  std::vector<wss::IBuffer> buffers_;
  size_t num_elements_;
  size_t element_size_in_bytes_;
};

}  // namespace _winml
