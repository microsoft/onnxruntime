
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <optional>

#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_shape.h"

#include "gsl/gsl-lite.hpp"

namespace onnxruntime {

class InitializerView {
 private:
  InitializerView(const InitializerView&) = delete;
  InitializerView& operator=(const InitializerView&) = delete;

 public:
  InitializerView() = default;

  static common::Status Create(
      const ONNX_NAMESPACE::TensorProto& tensor_proto, std::optional<InitializerView>& initializer);

  const TensorShape& Shape() const {
    return shape_;
  }

  int32_t Type() {
    return data_type_;
  }

  template <class T>
  gsl::span<const T> DataAsSpan() const {
    // make sure data bytes are more than sizeof (T)";
    return gsl::make_span(data<T>(), unpacked_tensor_.size() / sizeof(T));
  }

 private:
  template <class T>
  const T* data() const {
    // make sure T has no qualifier
    static_assert(std::is_same_v<typename std::decay<T>::type, T> &&
                  std::is_same_v<typename std::remove_pointer<T>::type, T>);
    return reinterpret_cast<const T*>(unpacked_tensor_.data());
  }

 private:
  TensorShape shape_;
  int32_t data_type_{0};
  std::vector<uint8_t> unpacked_tensor_;
};
}  // namespace onnxruntime
