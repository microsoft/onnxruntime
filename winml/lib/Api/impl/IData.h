// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "IEngine.h"

namespace _winml {

struct idata {
  virtual ~idata() {}

  virtual size_t num_elements() = 0;
  virtual size_t size_in_bytes() = 0;
  virtual size_t num_buffers() = 0;
  virtual std::vector<wss::IBuffer>& buffers() = 0;
  virtual gsl::span<byte> buffer(bool should_sync_buffer) = 0;
  virtual bool flush() = 0;
  virtual void set(size_t data_size, const byte* data) = 0;
};

}  // namespace _winml
