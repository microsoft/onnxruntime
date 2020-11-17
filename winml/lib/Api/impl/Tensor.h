// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DataBufferContainer.h"
#include "StringBufferContainer.h"

//
// the Tensor class is the actual object for CPU memory buffers.
// TensorBase contains one of these to represent the raw memory
// GetCpuResource() returns it
//
namespace _winml {

template <T>
struct buffer_container {
  using Type = data_buffer_container;
}

template <>
struct buffer_container<std::string> {
  using Type = string_buffer_container;
}

size_t compute_size_of_shape(const std::vector<int64_t>& shape) {
  auto size_of_shape =
    static_cast<size_t>(
      std::accumulate(
        std::begin(shape),
        std::end(shape),
        static_cast<int64_t>(1),
        std::multiplies<int64_t>()));
  return size_of_shape;
}

template <typename T>
auto create_tensor_buffer(
  using buffer_container = typename buffer_container<T>::Type;
  const std::vector<int64_t>& shape,
  const wfc::IIterable<wss::IBuffer>& buffers) {
    return buffer_container::create(compute_size_of_shape(shape), sizeof(T), buffers); 
}

template <>
auto create_tensor_buffer<std::string>(
  using buffer_container = typename buffer_container<T>::Type;
  const std::vector<int64_t>& shape,
  const wfc::IIterable<wss::IBuffer>& /*buffers*/) {
    return buffer_container::Create(compute_size_of_shape(shape)); 
}

template <typename T>
class Tensor {
 private:
  using buffer_container = typename buffer_container<T>::Type;
  std::shared_ptr<buffer_container> buffer_;
  std::vector<int64_t> shape_;

 private:
  Tensor() = delete;

 public:
  template <typename T>
  Tensor(std::vector<int64_t> const& shape, wfc::IIterable<wss::IBuffer> const& buffers) :
    shape_(shape),
    buffer_(create_tensor_buffer<T>(shape, buffers)) {}

  template <typename T>
  Tensor(const std::vector<int64_t>& shape) :
    shape_(shape),
    buffer_(create_tensor_buffer<T>(shape, nullptr)) {}

  template <typename T>
  Tensor(const std::vector<int64_t>&& shape) :
    shape_(std::move(shape)),
    buffer_(create_tensor_buffer<T>(shape, nullptr)) {}

  auto number_of_elements() const {
    return buffer_->num_elements();
  }

  auto size_in_bytes() const {
    return buffer_->size_in_bytes();
  }

  auto num_buffers() {
    return buffer_->num_buffers();
  }

  auto& buffers() {
    return buffer_->buffers();
  }

  auto buffer(bool should_sync_buffer = true) {
    auto span = buffer_->buffer(should_sync_buffer);
    return gsl::span<T>(reinterpret_cast<T*>(span.data()), buffer_->num_elements());
  }

  auto flush() {
    return buffer_->flush();
  }

  void set(size_t size, const T* data) {
    buffer_->set(size * sizeof(T), data);
  }

  void set(std::vector<T>&& other) {
    buffer_->set<T>(other);
  }

  const std::vector<int64_t>& shape() const {
    return shape_;
  }

  auto get_tensor_buffer() {
    return buffer_;
  }
};
}  // namespace _winml