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
  VectorBuffer(size_t size) : m_buffer(size) {}

  ~VectorBuffer() {}

  uint32_t Capacity() const {
    return static_cast<uint32_t>(m_buffer.size());
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
    *value = m_buffer.data();
    return S_OK;
  }

 private:
  std::vector<BYTE> m_buffer;
};

template <typename T>
class TensorBuffer {
  std::vector<wss::IBuffer> buffers_;
  size_t size_;

  TensorBuffer(size_t size) : size_(size), buffers_(1) {
    buffers_[0] = winrt::make<VectorBuffer>(size * sizeof(T));
    
    auto buffer = Buffer(0);

    // The initial release of WinML (RS5) shipped with behavior that would
    // zero-initialize uninitialized tensors. After measuring, the performance impact
    // of memsetting the memory buffer is quite small (<1ms for 3channel 720x720 TensorFloats).
    // To maintain parity with RS5 behavior, we always zero out the memory buffer.
    memset(buffer.second, 0, buffer.first);
  }

  TensorBuffer(
      size_t size,
      wfc::IIterable<wss::IBuffer> const& buffers) :
        size_(size),
        buffers_(begin(buffers), end(buffers)) { }

 public:
  typedef std::shared_ptr<TensorBuffer> TensorBufferPtr;

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

  // this is the count of elements
  auto Size() {
    return size_;
  }

  // this is the size in bytes
  auto SizeInBytes() {
    return size_ * sizeof(T);
  }

  auto Buffer(size_t index) {
    size_t size =
        buffers_.size() == 1 ?
        size_ :
        static_cast<size_t>(buffers_[index].Capacity());

    T* current_data = nullptr;
    auto bufferByteAccess = buffers_[0].as<Windows::Storage::Streams::IBufferByteAccess>();
    bufferByteAccess->Buffer(reinterpret_cast<BYTE**>(&current_data));
    return std::make_pair(size, current_data);
  }

  auto Buffer() {
    if (buffers_.size() == 1) {
      auto pair = Buffer(0);
      using Resource = std::unique_ptr<void, std::function<void(void*)>>;
      return std::make_pair(size_, Resource(pair.second, [](void*){}));
    }

    size_t start = 0;

    T* raw_buffer = new T[size_];
    for (size_t i = 0; i < buffers_.size() && start < size_; i++) {
      auto pair = Buffer(i);
      auto current_size = pair.first;
      auto current_buffer = pair.second;

      if (start + current_size > size_) {
        current_size = size_ - start;
      }

      memcpy(raw_buffer + start, current_buffer, current_size);
      start += current_size;
    }

    return std::make_pair(size_, Resource(raw_buffer, [](void* data) { delete[] data; }));
  }

  auto Set(size_t size, const T* data) {
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        size <= size_,
        "Argument size (%llu) exceeds the tensor size (%llu).",
        size,
        size_);

    
    size_t start = 0;
    for (size_t i = 0; i < buffers_.size() && size > start; i++) {
      auto pair = Buffer(i);
      auto current_size = pair.first;
      auto current_buffer = pair.second;

      if (size - start < current_size) {
        current_size = size - start;
      }

      memcpy(current_buffer, data + start, current_size);
      start += current_size;
    }
  }

  auto Set(std::vector<T>&& moveableData) {
    Set(moveableData.size(), moveableData.data());
  }
};

template <>
class TensorBuffer<std::string> {
  std::vector<std::string> m_buffer;

  TensorBuffer(size_t size) : m_buffer(size) {}

 public:
  typedef std::shared_ptr<TensorBuffer> TensorBufferPtr;

  static auto Create(size_t size) {
    return std::shared_ptr<TensorBuffer>(new TensorBuffer(size));
  }

  auto Size() {
    return m_buffer.size();
  }

  // this is the size in bytes
  auto SizeInBytes() {
    return m_buffer.size();
  }

  auto Buffer() {
    auto size = m_buffer.size();
    using Resource = std::unique_ptr<void, std::function<void(void*)>>;
    return std::make_pair(size, Resource(new std::string[size], [](void* data) { delete[] data; }));
  }

  auto Set(size_t size, std::string_view* data) {
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        size <= m_buffer.size(),
        "Argument size (%d) exceeds the tensor size (%d).",
        static_cast<int>(size),
        static_cast<int>(m_buffer.size()));

    // Copy
    std::copy(data, data + size, m_buffer.begin());
  }
};
}  // namespace _winml