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
  wss::IBuffer m_buffer;
  uint32_t m_size;

  TensorBuffer(uint32_t size) : m_size(size),
                                m_buffer(winrt::make<VectorBuffer>(size * sizeof(T))) {
    auto buffer = Buffer();

    // The initial release of WinML (RS5) shipped with behavior that would
    // zero-initialize uninitialized tensors. After measuring, the performance impact
    // of memsetting the memory buffer is quite small (<1ms for 3channel 720x720 TensorFloats).
    // To maintain parity with RS5 behavior, we always zero out the memory buffer.
    memset(buffer.second, 0, buffer.first);
  }

  TensorBuffer(
      uint32_t size,
      wss::IBuffer buffer) : m_size(size),
                                                          m_buffer(buffer) {}

 public:
  typedef std::shared_ptr<TensorBuffer> TensorBufferPtr;

  static auto Create(uint32_t size) {
    return std::shared_ptr<TensorBuffer>(new TensorBuffer(size));
  }

  static auto Create(
      uint32_t size,
      wss::IBuffer buffer) {
    return std::shared_ptr<TensorBuffer>(new TensorBuffer(size, buffer));
  }

  // this is the count of elements
  auto Size() {
    return m_size;
  }

  // this is the size in bytes
  auto SizeInBytes() {
    return m_size * sizeof(T);
  }

  auto Buffer() {
    T* pData;
    auto bufferByteAccess = m_buffer.as<Windows::Storage::Streams::IBufferByteAccess>();
    bufferByteAccess->Buffer(reinterpret_cast<BYTE**>(&pData));

    return std::make_pair(m_size, pData);
  }

  auto Set(uint32_t size, const T* pData) {
    WINML_THROW_HR_IF_FALSE_MSG(
        E_INVALIDARG,
        size <= m_size,
        "Argument size (%u) exceeds the tensor size (%u).",
        size,
        m_size);

    memcpy(Buffer().second, pData, m_buffer.Capacity());
  }

  auto Set(std::vector<T>&& moveableData) {
    Set(moveableData.size(), moveableData.data());
  }
};

template <>
class TensorBuffer<std::string> {
  std::vector<std::string> m_buffer;

  TensorBuffer(uint32_t size) : m_buffer(size) {}

 public:
  typedef std::shared_ptr<TensorBuffer> TensorBufferPtr;

  static auto Create(uint32_t size) {
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
    return std::make_pair(gsl::narrow_cast<uint32_t>(m_buffer.size()), m_buffer.data());
  }

  auto Set(uint32_t size, std::string_view* data) {
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