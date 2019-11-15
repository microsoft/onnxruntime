// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "TensorBuffer.h"
#include "MLValueHelpers.h"

//
// the Tensor class is the actual object for CPU memory buffers.
// TensorBase contains one of these to represent the raw memory
// GetCpuResource() returns it
//
namespace Windows::AI::MachineLearning {
template <typename T>
class Tensor {
 private:
  using TensorBuffer = TensorBuffer<T>;
  using TensorBufferPtr = typename TensorBuffer::TensorBufferPtr;

  TensorBufferPtr m_buffer;
  std::vector<int64_t> m_shape;
  winrt::com_ptr<_winmla::IWinMLAdapter> adapter_;


 public:
  Tensor() = delete;

  Tensor(
      _winmla::IWinMLAdapter* adapter,
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
                                                                  buffer)) {
      adapter_.copy_from(adapter);
  }

  Tensor(
      _winmla::IWinMLAdapter* adapter,
      std::vector<int64_t> const& shape) : m_shape(shape),
                                           m_buffer(
                                               TensorBuffer::Create(
                                                   static_cast<uint32_t>(
                                                       std::accumulate(
                                                           std::begin(shape),
                                                           std::end(shape),
                                                           static_cast<int64_t>(1),
                                                           std::multiplies<int64_t>())))) {
      adapter_.copy_from(adapter);
  }

  Tensor(
      _winmla::IWinMLAdapter* adapter,
      std::vector<int64_t> const&& shape) : m_shape(std::move(shape)),
                                            m_buffer(
                                                TensorBuffer::Create(
                                                    static_cast<uint32_t>(
                                                        std::accumulate(
                                                            std::begin(shape),
                                                            std::end(shape),
                                                            static_cast<int64_t>(1),
                                                            std::multiplies<int64_t>())))) {
      adapter_.copy_from(adapter);
  }

  auto size() const {
    return m_buffer->Size();
  }

  auto buffer() {
    return m_buffer->Buffer();
  }

  _winmla::IOrtValue* GetValue() {
    // Get the data type
    auto type = adapter_->GetTensorType(TensorKindFrom<T>::Type);
    // create the ml value
    winrt::com_ptr<_winmla::IOrtValue> value;
    WINML_THROW_IF_FAILED(adapter_->CreateCPUMLValue(&m_shape, type, buffer().second, value.put()));
    return value.detach();
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