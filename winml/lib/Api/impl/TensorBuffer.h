// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "robuffer.h"
#include "winrt/Windows.Storage.Streams.h"

namespace _winml {

class VectorBuffer : public winrt::implements<
                         VectorBuffer,
                         wss::IBuffer,
                         Windows::Storage::Streams::IBufferByteAccess> {
 public:
  VectorBuffer(size_t size) : buffer_(size) {}

  uint32_t Capacity() const {
    return static_cast<uint32_t>(buffer_.size());
  }

  uint32_t Length() const {
    throw winrt::hresult_error(E_NOTIMPL);
  }

  void Length(uint32_t /*value*/) {
    throw winrt::hresult_error(E_NOTIMPL);
  }

  STDMETHOD(Buffer)
  (uint8_t** value) {
    RETURN_HR_IF_NULL(E_POINTER, value);
    *value = buffer_.data();
    return S_OK;
  }

 private:
  std::vector<BYTE> buffer_;
};

template <typename T>
class TensorBuffer {
  wss::IBuffer combined_buffer_;
  std::vector<wss::IBuffer> buffers_;
  size_t size_;

  TensorBuffer(size_t size) :
      size_(size),
      combined_buffer_(winrt::make<VectorBuffer>(size * sizeof(T))),
      buffers_ { combined_buffer_ } {
    auto buffer = BufferAt(0);

    // The initial release of WinML (RS5) shipped with behavior that would
    // zero-initialize uninitialized tensors. After measuring, the performance impact
    // of memsetting the memory buffer is quite small (<1ms for 3channel 720x720 TensorFloats).
    // To maintain parity with RS5 behavior, we always zero out the memory buffer.
    memset(buffer.second, 0, buffer.first);
  }

  TensorBuffer(
      size_t size,
      wfc::IIterable<wss::IBuffer> const& buffers) : size_(size),
                                                     combined_buffer_(nullptr),
                                                     buffers_(begin(buffers), end(buffers)) {
    if (buffers_.size() == 1) {
      combined_buffer_ = buffers_[0];
    } else {
      // If there are many buffers, then the combined buffer will be a separately allocated value that combines all of the buffers.
      // This needs to be lazily done however, as the extra memory should not be allocated when not needed (GPU).
    }
  }

  auto CombinedBuffer() {
    if (combined_buffer_ == nullptr) {
      combined_buffer_ = winrt::make<VectorBuffer>(size_ * sizeof(T));
    }
    return BufferFrom(combined_buffer_);
  }

 public:
  static auto Create(size_t size) {
    return std::shared_ptr<TensorBuffer>(new TensorBuffer(size));
  }

  static auto Create(
      size_t size,
      wss::IBuffer buffer) {
    return std::shared_ptr<TensorBuffer>(new TensorBuffer(size, buffer));
  }

  static auto Create(
      size_t size,
      wfc::IIterable<wss::IBuffer> const& buffers) {
    return std::shared_ptr<TensorBuffer>(new TensorBuffer(size, buffers));
  }

  // this is the size in bytes
  auto SizeInBytes() {
    return size_ * sizeof(T);
  }

  auto NumBuffers() {
    return buffers_.size();
  }

  auto& Buffers() {
    return buffers_;
  }

  auto Buffer(bool should_sync_buffer) {
    if (buffers_.size() == 1) {
      // Single buffer optimization to not create a temporary buffer that concatenates disjoint buffers into one.
      return BufferAt(0);
    }

    auto combined_buffer = CombinedBuffer();
    if (should_sync_buffer) {
      auto size_in_bytes = combined_buffer.first;
      size_t offset_in_bytes = 0;
      for (size_t i = 0; i < buffers_.size() && offset_in_bytes < size_in_bytes; i++) {
        size_t current_size_in_bytes;
        T* current_buffer;
        std::tie(current_size_in_bytes, current_buffer) = BufferAt(i);

        if (size_in_bytes - offset_in_bytes < current_size_in_bytes) {
          current_size_in_bytes = size_in_bytes - offset_in_bytes;
        }

        auto buffer_start = reinterpret_cast<BYTE*>(combined_buffer.second) + offset_in_bytes;
        memcpy(buffer_start, current_buffer, current_size_in_bytes);
        offset_in_bytes += current_size_in_bytes;
      }
    }

    return combined_buffer;
  }

  auto Flush() {
    auto should_flush = buffers_.size() != 1;
    if (should_flush) {
      auto combined_buffer = CombinedBuffer();
      Set(combined_buffer.first, combined_buffer.second);
    }
    return should_flush;
  }

  auto Set(size_t size_in_bytes, const T* data) {
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        size_in_bytes <= (size_ * sizeof(T)),
        "Argument size (%llu) exceeds the tensor size (%llu).",
        size_in_bytes,
        size_ * sizeof(T));
    
    size_t offset_in_bytes = 0;
    for (size_t i = 0; i < buffers_.size() && size_in_bytes > offset_in_bytes; i++) {
      size_t current_size_in_bytes;
      T* current_buffer;
      std::tie(current_size_in_bytes, current_buffer) = BufferAt(i);

      if (size_in_bytes - offset_in_bytes < current_size_in_bytes) {
        current_size_in_bytes = size_in_bytes - offset_in_bytes;
      }

      memcpy(current_buffer, reinterpret_cast<const BYTE*>(data) + offset_in_bytes, current_size_in_bytes);
      offset_in_bytes += current_size_in_bytes;
    }
  }

  auto Set(std::vector<T>&& moveableData) {
    Set(moveableData.size(), moveableData.data());
  }

 private:
  auto BufferFrom(wss::IBuffer buffer) {
    T* current_data = nullptr;
    auto bufferByteAccess = buffer.as<Windows::Storage::Streams::IBufferByteAccess>();
    bufferByteAccess->Buffer(reinterpret_cast<BYTE**>(&current_data));
    return std::make_pair(
        static_cast<size_t>(buffer.Capacity()),
        current_data);
  }

  auto BufferAt(size_t index) {
    return BufferFrom(buffers_[index]);
  }
};

template <>
class TensorBuffer<std::string> {
  std::vector<std::string> buffer_;

  TensorBuffer(size_t size) : buffer_(size) {}

 public:
  static auto Create(size_t size) {
    return std::shared_ptr<TensorBuffer>(new TensorBuffer(size));
  }

  // this is the size in bytes
  auto SizeInBytes() {
    return buffer_.size();
  }

  auto NumBuffers() {
    return 1;
  }

  auto Flush() {
    return false;
  }

  auto Buffers() -> std::vector<wss::IBuffer>& {
    WINML_THROW_HR(E_UNEXPECTED);
  }

  auto BufferAt(size_t index) {
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        index == 0,
        "TensorString can only be backed by a single buffer!");
    return std::make_pair(buffer_.size(), buffer_.data());
  }

  auto Buffer(bool /*should_sync_buffer*/) {
    return std::make_pair(buffer_.size(), buffer_.data());
  }

  auto Set(size_t size, std::string_view* data) {
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        size <= buffer_.size(),
        "Argument size (%d) exceeds the tensor size (%d).",
        static_cast<int>(size),
        static_cast<int>(buffer_.size()));

    // Copy
    std::copy(data, data + size, buffer_.begin());
  }
};
}  // namespace _winml