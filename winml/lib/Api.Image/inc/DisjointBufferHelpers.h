// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>

namespace _winml {

void LoadSpanFromDisjointBuffers(
  size_t num_buffers, std::function<gsl::span<byte>(size_t)> get_buffer, gsl::span<byte>& buffer_span
);

void StoreSpanIntoDisjointBuffers(
  size_t num_buffers, std::function<gsl::span<byte>(size_t)> get_buffer, gsl::span<byte>& buffer_span
);

}  // namespace _winml
