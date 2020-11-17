// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "pch.h"

#include "DataBufferContainer.h"
#include "VectorBackedBuffer.h"
#include "robuffer.h"
#include "winrt/Windows.Storage.Streams.h"
#include "DisjointBufferHelpers.h"

namespace _winml {

std::shared_ptr<data_buffer_container> data_buffer_container::create(
  size_t num_elements,
  size_t element_size_in_bytes,
  wfc::IIterable<wss::IBuffer> const& buffers) {
  return std::make_shared<data_buffer_container>(num_elements, element_size_in_bytes, buffers));
}

data_buffer_container::data_buffer_container(
  size_t num_elements, size_t element_size_in_bytes, wfc::IIterable<wss::IBuffer> const& buffers) :
  num_elements_(num_elements),
  element_size_in_bytes_(element_size_in_bytes),
  combined_buffer_(nullptr),
  buffers_() {
  if (buffers != nullptr) {
    buffers_ = { begin(buffers), end(buffers) };
  }
  
  if (buffers_.size() == 0) {
    combined_buffer_ = winrt::make<vector_backed_buffer>(num_elements * element_size_in_bytes);
    buffers_ = { combined_buffer_ };
    auto buffer = buffer_at(0);
  
    // The initial release of WinML (RS5) shipped with behavior that would
    // zero-initialize uninitialized tensors. After measuring, the performance impact
    // of memsetting the memory buffer is quite small (<1ms for 3channel 720x720 TensorFloats).
    // To maintain parity with RS5 behavior, we always zero out the memory buffer.
    memset(buffer.data(), 0, buffer.size_bytes());
  }
  else if (buffers_.size() == 1) {
    combined_buffer_ = buffers_[0];
  }
  else {
    // If there are many buffers, then the combined buffer will be a separately allocated value that combines all of the buffers.
    // This needs to be lazily done however, as the extra memory should not be allocated when not needed (GPU).
  }
}

size_t data_buffer_container::num_elements() {
  return num_elements_;
}

size_t data_buffer_container::size_in_bytes() {
  return num_elements_ * element_size_in_bytes_;
}

size_t data_buffer_container::num_buffers() {
  return buffers_.size();
}

std::vector<wss::IBuffer>& data_buffer_container::buffers() {
  return buffers_;
}

gsl::span<byte> data_buffer_container::buffer(bool should_sync_buffer) {
  if (buffers_.size() == 1) {
    // Single buffer optimization to not create a temporary buffer that concatenates disjoint buffers into one.
    return buffer_at(0);
  }
  auto span = combined_buffer();
  if (should_sync_buffer) {
    _winml::LoadSpanFromDisjointBuffers(
      buffers_.size(),
      [this](size_t i) { return buffer_at(i); },
      span);
  }

  return span;
}

bool data_buffer_container::flush() {
  auto should_flush = buffers_.size() != 1;
  if (should_flush) {
    auto span = combined_buffer();
    _winml::LoadSpanFromDisjointBuffers(
        buffers_.size(),
        [this](size_t i) { return buffer_at(i); },
        span);
  }
  return should_flush;
}

void data_buffer_container::set(size_t size_in_bytes, const T* data) {
  WINML_THROW_HR_IF_FALSE_MSG(
      E_INVALIDARG,
      size_in_bytes <= (num_elements_ * element_size_in_bytes_),
      "Argument size (%llu) exceeds the tensor size (%llu).",
      static_cast<uint64_t>(size_in_bytes),
      static_cast<uint64_t>(num_elements_ * element_size_in_bytes_));
  
  gsl::span<byte> span(reinterpret_cast<byte*>(const_cast<T*>(data)), size_in_bytes);
  _winml::LoadSpanFromDisjointBuffers(
    buffers_.size(),
    [this](size_t i) { return buffer_at(i); },
    span);
}

static gsl::span<byte> get_span_from_ibuffer(wss::IBuffer buffer) {
  byte* current_data = nullptr;
  auto bufferByteAccess = buffer.as<Windows::Storage::Streams::IBufferByteAccess>();
  bufferByteAccess->Buffer(&current_data);
  return gsl::span<byte>(
      current_data,
      static_cast<size_t>(buffer.Capacity()));
}

gsl::span<byte> data_buffer_container::buffer_at(size_t index) {
  return get_span_from_ibuffer(buffers_[index]);
}

gsl::span<byte> data_buffer_container::combined_buffer() {
  if (combined_buffer_ == nullptr) {
    combined_buffer_ = winrt::make<vector_backed_buffer>(num_elements_ * element_size_in_bytes_);
  }
  return get_span_from_ibuffer(combined_buffer_);
}

}  // namespace _winml