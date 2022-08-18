
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
 public:
  InitializerView() = default;
  InitializerView(const ONNX_NAMESPACE::TensorProto& tensor_proto);

  common::Status Create(
      const ONNX_NAMESPACE::TensorProto& tensor_proto);

  const TensorShape& Shape() const {
    return shape_;
  }

  int32_t GetElementType() {
    return dtype_->GetDataType();
  }

  template <class T>
  gsl::span<const T> DataAsSpan() const {
    ORT_ENFORCE(utils::IsPrimitiveDataType<T>(dtype_), "Data type mismatch. ",
                "T ", "!=", dtype_);
    return gsl::make_span(data<T>(), unpacked_tensor_.size() / sizeof(T));
  }

  size_t SizeInBytes() {
    return unpacked_tensor_.size();
  }

  template <class T = char>
  gsl::span<const T> DataAsByteSpan() const {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, char>,
                  "expected int8|uint8|char");
    return gsl::make_span(data<T>(), unpacked_tensor_.size() / sizeof(T));
  }

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(InitializerView);

  template <class T>
  const T* data() const {
    // make sure T has no qualifier
    static_assert(std::is_same_v<typename std::decay<T>::type, T> &&
                  std::is_same_v<typename std::remove_pointer<T>::type, T>);
    return reinterpret_cast<const T*>(unpacked_tensor_.data());
  }

 private:
  TensorShape shape_;
  const PrimitiveDataTypeBase* dtype_{nullptr};
  std::vector<uint8_t> unpacked_tensor_;
};
}  // namespace onnxruntime
