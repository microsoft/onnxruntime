// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "TensorBuffer.h"

// we further specialize these base types for a couple of extra tensor element types
namespace Ort {
template <>
struct TypeToTensorType<std::string> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING; };
template <>
struct TypeToTensorType<onnxruntime::MLFloat16> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}

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
  std::vector<int64_t> shape_;

 public:
  Tensor() = delete;

  Tensor(
      std::vector<int64_t> const& shape,
      winrt::Windows::Storage::Streams::IBuffer buffer) : shape_(shape),
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

  auto buffer() {
    return m_buffer->Buffer();
  }

  Ort::Value GetValue() {
    // this is cpu memory
    // TODO:  what is the difference between the device allocator and the arena allocator?
    Ort::MemoryInfo cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // create the OrtValue as a tensor letting ort know that we own the data buffer
    auto value = Ort::Value::CreateTensor<T>(
        cpu_memory,
        buffer().second,
        m_buffer->SizeInBytes(),
        shape_.data(),
        shape_.size());
//        Ort::TypeToTensorType<T>::type);
    return value;
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
};
}  // namespace Windows::AI::MachineLearning