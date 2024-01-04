// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "lib/Api/pch/pch.h"

#include "VectorBackedBuffer.h"

namespace _winml {

vector_backed_buffer::vector_backed_buffer(size_t size) : buffer_(size) {
}

uint32_t vector_backed_buffer::Capacity() const {
  return static_cast<uint32_t>(buffer_.size());
}

uint32_t vector_backed_buffer::Length() const {
  throw winrt::hresult_error(E_NOTIMPL);
}

void vector_backed_buffer::Length(uint32_t /*value*/) {
  throw winrt::hresult_error(E_NOTIMPL);
}

STDMETHODIMP vector_backed_buffer::Buffer(uint8_t** value) {
  RETURN_HR_IF_NULL(E_POINTER, value);
  *value = buffer_.data();
  return S_OK;
}

}  // namespace _winml
