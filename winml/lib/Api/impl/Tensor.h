// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "NumericData.h"
#include "StringData.h"

//
// the Tensor class is the actual object for CPU memory buffers.
// TensorBase contains one of these to represent the raw memory
// GetCpuResource() returns it
//
namespace _winml {

inline size_t compute_size_of_shape(const std::vector<int64_t>& shape) {
  auto size_of_shape = static_cast<size_t>(
    std::accumulate(std::begin(shape), std::end(shape), static_cast<int64_t>(1), std::multiplies<int64_t>())
  );
  return size_of_shape;
}

template <typename T>
inline auto create_data(const std::vector<int64_t>& shape, const wfc::IIterable<wss::IBuffer>& buffers) {
  return _winml::numeric_data::create(compute_size_of_shape(shape), sizeof(T), buffers);
}

template <>
inline auto
create_data<std::string>(const std::vector<int64_t>& shape, const wfc::IIterable<wss::IBuffer>& /*buffers*/) {
  return _winml::string_data::create(compute_size_of_shape(shape));
}

template <typename T>
class Tensor {
 private:
  std::shared_ptr<_winml::idata> data_;
  std::vector<int64_t> shape_;

 private:
  Tensor() = delete;

 public:
  Tensor(const std::vector<int64_t>& shape) : data_(create_data<T>(shape, nullptr)), shape_(shape) {}

  Tensor(const std::vector<int64_t>& shape, const wfc::IIterable<wss::IBuffer>& buffers)
    : data_(create_data<T>(shape, buffers)),
      shape_(shape) {}

  auto size_in_bytes() const { return data_->size_in_bytes(); }

  auto num_buffers() { return data_->num_buffers(); }

  auto& buffers() { return data_->buffers(); }

  gsl::span<T> buffer(bool should_sync_buffer = true) {
    auto span = data_->buffer(should_sync_buffer);
    return gsl::span<T>(reinterpret_cast<T*>(span.data()), data_->num_elements());
  }

  auto flush() { return data_->flush(); }

  void set(size_t size, const T* data) {
    auto size_in_bytes = size * sizeof(T);
    data_->set(size_in_bytes, reinterpret_cast<const byte*>(data));
  }

  const std::vector<int64_t>& shape() const { return shape_; }

  auto get_data() { return data_; }
};
}  // namespace _winml
