// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "TensorBuffer.h"

//
// the Tensor class is the actual object for CPU memory buffers.
// TensorBase contains one of these to represent the raw memory
// GetCpuResource() returns it
//
namespace _winml {

template <typename T>
class Tensor {
 private:
  using TensorBuffer = TensorBuffer<T>;
  using TensorBufferPtr = typename TensorBuffer::TensorBufferPtr;

  TensorBufferPtr m_buffer;
  std::vector<int64_t> shape_;

 public:
  Tensor() = delete;

  Tensor(
      std::vector<int64_t> const& shape,
      wss::IBuffer buffer) : shape_(shape),
                             m_buffer(
                                 TensorBuffer::Create(
                                     static_cast<uint32_t>(
                                         std::accumulate(
                                             std::begin(shape),
                                             std::end(shape),
                                             static_cast<int64_t>(1),
                                             std::multiplies<int64_t>())),
                                     buffer)) {
  }

  Tensor(
      std::vector<int64_t> const& shape) : shape_(shape),
                                           m_buffer(
                                               TensorBuffer::Create(
                                                   static_cast<uint32_t>(
                                                       std::accumulate(
                                                           std::begin(shape),
                                                           std::end(shape),
                                                           static_cast<int64_t>(1),
                                                           std::multiplies<int64_t>())))) {
  }

  Tensor(
      std::vector<int64_t> const&& shape) : shape_(std::move(shape)),
                                            m_buffer(
                                                TensorBuffer::Create(
                                                    static_cast<uint32_t>(
                                                        std::accumulate(
                                                            std::begin(shape),
                                                            std::end(shape),
                                                            static_cast<int64_t>(1),
                                                            std::multiplies<int64_t>())))) {
  }

  auto size() const {
    return m_buffer->Size();
  }

  auto size_in_bytes() const {
    return m_buffer->SizeInBytes();
  }

  auto buffer() {
    return m_buffer->Buffer();
  }

  void set(uint32_t size, const T* pData) {
    m_buffer->Set(size, pData);
  }

  void set(std::vector<T>&& other) {
    m_buffer->Set(other);
  }

  const std::vector<int64_t>& shape() const {
    return shape_;
  }

  auto get_tensor_buffer() {
    return m_buffer;
  }
};
}  // namespace _winml