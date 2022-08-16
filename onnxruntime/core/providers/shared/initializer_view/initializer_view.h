
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>

#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_shape.h"

#include "gsl/gsl-lite.hpp"

namespace onnxruntime {

class InitializerView {
 private:
  InitializerView(const InitializerView&) = delete;
  InitializerView& operator=(const InitializerView&) = delete;
  InitializerView() = default;

 public:
  static std::unique_ptr<InitializerView> Create(
      const ONNX_NAMESPACE::TensorProto& tensor_proto);
  const TensorShape& Shape() {
    return shape_;
  }

  const DataTypeImpl* Type() {
    return type_;
  }

  template <class T>
  gsl::span<const T> DataAsSpan() {
    ORT_ENFORCE(unpacked_tensor_.size() >= sizeof(T), " data bytes must more than sizeof (T)");
    return gsl::make_span(data<T>(), unpacked_tensor_.size() / sizeof(T));
  }

 private:
  template <class T>
  const T* data() {
    // make sure T has no qualifier
    static_assert(std::is_same_v<typename std::decay<T>::type, T> &&
                  std::is_same_v<typename std::remove_pointer<T>::type, T>);
    return reinterpret_cast<const T*>(unpacked_tensor_.data());
  }

 private:
  TensorShape shape_;
  const DataTypeImpl* type_{nullptr};
  std::vector<uint8_t> unpacked_tensor_;
};
}  // namespace onnxruntime
