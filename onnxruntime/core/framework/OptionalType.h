// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include "core/framework/TensorSeq.h"

namespace onnxruntime {
class OptionalType {
 public:
  OptionalType() = default;

  explicit OptionalType(MLDataType type) noexcept {
    SetType(type);
  }

  explicit OptionalType(Tensor tensor) noexcept {
    SetType(elem_type);
  }

  void SetType(MLDataType type) {
    ORT_ENFORCE(elem_type_ != nullptr, "Tensor sequence must contain only primitive types");	  
    type_ = type;
  }

  MLDataType DataType() const noexcept { return type_; }

 private:
  const MLDataType type_{};

  void* element_;
};

}  // namespace onnxruntime
