// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "TensorBuffer.h"
#include "MLValueHelpers.h"

namespace Windows::AI::MachineLearning {
template <typename T>
class Tensor {
 private:
  using TensorBuffer = TensorBuffer<T>;
  using TensorBufferPtr = typename TensorBuffer::TensorBufferPtr;

  TensorBufferPtr m_buffer;
  std::vector<int64_t> m_shape;

 public:
  Tensor() = delete;

  Tensor(
      std::vector<int64_t> const& shape,
      winrt::Windows::Storage::Streams::IBuffer buffer) : m_shape(shape),
                                                          m_buffer(
                                                              TensorBuffer::Create(
                                                                  static_cast<uint32_t>(
                                                                      std::accumulate(
                                                                          std::begin(shape),
                                                                          std::end(shape),
                                                                          static_cast<int64_t>(1),
                                                                          std::multiplies<int64_t>())),
                                                                  buffer)) {}

  Tensor(std::vector<int64_t> const& shape) : m_shape(shape),
                                              m_buffer(
                                                  TensorBuffer::Create(
                                                      static_cast<uint32_t>(
                                                          std::accumulate(
                                                              std::begin(shape),
                                                              std::end(shape),
                                                              static_cast<int64_t>(1),
                                                              std::multiplies<int64_t>())))) {}

  Tensor(std::vector<int64_t> const&& shape) : m_shape(std::move(shape)),
                                               m_buffer(
                                                   TensorBuffer::Create(
                                                       static_cast<uint32_t>(
                                                           std::accumulate(
                                                               std::begin(shape),
                                                               std::end(shape),
                                                               static_cast<int64_t>(1),
                                                               std::multiplies<int64_t>())))) {}

  auto size() const {
    return m_buffer->Size();
  }

  auto buffer() {
    return m_buffer->Buffer();
  }

  OrtValue MLValue() {
    // Get the shape
    onnxruntime::TensorShape shape(m_shape);
    // Get the data type
    auto type = onnxruntime::DataTypeImpl::GetType<T>();

    return MLValueHelpers::CreateMLValue(shape, type, buffer().second);
  }

  void set(uint32_t size, const T* pData) {
    m_buffer->Set(size, pData);
  }

  void set(std::vector<T>&& other) {
    m_buffer->Set(other);
  }

  const std::vector<int64_t>& shape() const {
    return m_shape;
  }
};
}  // namespace Windows::AI::MachineLearning