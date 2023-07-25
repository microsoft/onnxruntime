// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "lib/Api/pch/pch.h"

#include "impl/NumericData.h"
#include "VectorBackedBuffer.h"
#include "robuffer.h"
#include "winrt/Windows.Storage.Streams.h"
#include "DisjointBufferHelpers.h"

namespace _winml {

std::shared_ptr<_winml::idata> numeric_data::create(
  size_t num_elements, size_t element_size_in_bytes, wfc::IIterable<wss::IBuffer> const& buffers
) {
  return std::make_shared<numeric_data>(num_elements, element_size_in_bytes, buffers);
}

numeric_data::numeric_data(
  size_t num_elements, size_t element_size_in_bytes, wfc::IIterable<wss::IBuffer> const& buffers
)
  : combined_buffer_(nullptr),
    buffers_(),
    num_elements_(num_elements),
    element_size_in_bytes_(element_size_in_bytes) {
  if (buffers != nullptr) {
    buffers_ = {begin(buffers), end(buffers)};
  }

  if (buffers_.size() == 0) {
    combined_buffer_ = winrt::make<vector_backed_buffer>(num_elements * element_size_in_bytes);
    buffers_ = {combined_buffer_};
    auto buffer = buffer_at(0);

      // The initial release of WinML (RS5) shipped with behavior that would
    // zero-initialize uninitialized tensors. After measuring, the performance impact
    // of memsetting the memory buffer is quite small (<1ms for 3channel 720x720 TensorFloats).
    // To maintain parity with RS5 behavior, we always zero out the memory buffer.
    memset(buffer.data(), 0, buffer.size_bytes());
  } else if (buffers_.size() == 1) {
    combined_buffer_ = buffers_[0];
  } else {
    // If there are many buffers, then the combined buffer will be a separately allocated value that combines all of the buffers.
    // This needs to be lazily done however, as the extra memory should not be allocated when not needed (GPU).
  }
}

size_t numeric_data::num_elements() {
  return num_elements_;
}

size_t numeric_data::size_in_bytes() {
  return num_elements_ * element_size_in_bytes_;
}

size_t numeric_data::num_buffers() {
  return buffers_.size();
}

std::vector<wss::IBuffer>& numeric_data::buffers() {
  return buffers_;
}

gsl::span<byte> numeric_data::buffer(bool should_sync_buffer) {
  if (buffers_.size() == 1) {
    // Single buffer optimization to not create a temporary buffer that concatenates disjoint buffers into one.
    return buffer_at(0);
  }
  auto span = combined_buffer();
  if (should_sync_buffer) {
    _winml::LoadSpanFromDisjointBuffers(
      buffers_.size(), [this](size_t i) { return buffer_at(i); }, span
    );
  }

  return span;
}

bool numeric_data::flush() {
  auto should_flush = buffers_.size() != 1;
  if (should_flush) {
    auto span = combined_buffer();
    _winml::StoreSpanIntoDisjointBuffers(
      buffers_.size(), [this](size_t i) { return buffer_at(i); }, span
    );
  }
  return should_flush;
}

void numeric_data::set(size_t data_size, const byte* data) {
  WINML_THROW_HR_IF_FALSE_MSG(
    E_INVALIDARG,
    data_size <= (num_elements_ * element_size_in_bytes_),
    "Argument size (%llu) exceeds the tensor size (%llu).",
    static_cast<uint64_t>(data_size),
    static_cast<uint64_t>(num_elements_ * element_size_in_bytes_)
  );

  gsl::span<byte> span(const_cast<byte*>(data), data_size);
  _winml::StoreSpanIntoDisjointBuffers(
    buffers_.size(), [this](size_t i) { return buffer_at(i); }, span
  );
}

static gsl::span<byte> get_span_from_ibuffer(wss::IBuffer buffer) {
  byte* current_data = nullptr;
  auto bufferByteAccess = buffer.as<Windows::Storage::Streams::IBufferByteAccess>();
  bufferByteAccess->Buffer(&current_data);
  return gsl::span<byte>(current_data, static_cast<size_t>(buffer.Capacity()));
}

gsl::span<byte> numeric_data::buffer_at(size_t index) {
  return get_span_from_ibuffer(buffers_[index]);
}

gsl::span<byte> numeric_data::combined_buffer() {
  if (combined_buffer_ == nullptr) {
    combined_buffer_ = winrt::make<vector_backed_buffer>(num_elements_ * element_size_in_bytes_);
  }
  return get_span_from_ibuffer(combined_buffer_);
}

}  // namespace _winml
