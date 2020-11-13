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
  std::shared_ptr<TensorBuffer<T>> buffer_;
  std::vector<int64_t> shape_;

 public:
  Tensor() = delete;

  Tensor(
      std::vector<int64_t> const& shape,
      wfc::IIterable<wss::IBuffer> const& buffers) :
                            shape_(shape),
                            buffer_(TensorBuffer<T>::Create(
                                        static_cast<size_t>(std::accumulate(
                                             std::begin(shape), std::end(shape),
                                             static_cast<int64_t>(1), std::multiplies<int64_t>())),
                                        buffers)) {}

  Tensor(
      std::vector<int64_t> const& shape) : shape_(shape),
                                           buffer_(TensorBuffer<T>::Create(
                                                        static_cast<size_t>(std::accumulate(
                                                            std::begin(shape), std::end(shape),
                                                            static_cast<int64_t>(1),
                                                            std::multiplies<int64_t>())))) {}

  Tensor(
      std::vector<int64_t> const&& shape) : shape_(std::move(shape)),
                                            buffer_(TensorBuffer<T>::Create(
                                                        static_cast<size_t>(std::accumulate(
                                                            std::begin(shape), std::end(shape),
                                                            static_cast<int64_t>(1),
                                                            std::multiplies<int64_t>())))) {
  }

  auto number_of_elements() const {
    return buffer_->NumElements();
  }

  auto size_in_bytes() const {
    return buffer_->SizeInBytes();
  }

  auto num_buffers() {
    return buffer_->NumBuffers();
  }

  auto& buffers() {
    return buffer_->Buffers();
  }

  auto buffer(bool should_sync_buffer = true) {
    auto span = buffer_->Buffer(should_sync_buffer);
    return gsl::span<T>(reinterpret_cast<T*>(span.data()), buffer_->NumElements());
  }

  auto flush() {
    return buffer_->Flush();
  }

  void set(size_t size, const T* pData) {
    buffer_->Set(size * sizeof(T), pData);
  }

  void set(std::vector<T>&& other) {
    buffer_->Set(other);
  }

  const std::vector<int64_t>& shape() const {
    return shape_;
  }

  auto get_tensor_buffer() {
    return buffer_;
  }
};
}  // namespace _winml