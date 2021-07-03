// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include "core/framework/TensorSeq.h"

namespace onnxruntime {

class OptionalValue final {
 public:
  // Don't want to be able to create an Optional Type not bound to
  // any types.
  // For now, we don't support sequence of optionals, so this
  // is not really needed anyway.
  OptionalValue() = delete;

  ~OptionalValue() = default;

  // For creating "empty" optional instances bound to a type
  OptionalValue(MLDataType type) {
    SetType(type);
  }

  OptionalValue(void* data, MLDataType type, DeleteFunc deleter) {
    SetType(type);
    data_.reset(data, deleter);
  }

  template <typename T>
  const T& Get() const {
    ORT_ENFORCE(DataTypeImpl::GetType<T>() == type_, DataTypeImpl::GetType<T>(), " != ", type_);
    ORT_ENFORCE(HasElement(), "This Optional type instance does not contain any data");
    return *static_cast<T*>(data_.get());
  }

  template <typename T>
  T* GetMutable() {
    ORT_ENFORCE(DataTypeImpl::GetType<T>() == type_, DataTypeImpl::GetType<T>(), " != ", type_);
    ORT_ENFORCE(HasElement(), "This Optional type instance does not contain any data");
    return static_cast<T*>(data_.get());
  }

  // For now, Optional types will be limited to wrapping Tensor and TensorSeq only
  bool IsTensor() const noexcept {
    return (type_ != nullptr && type_->IsTensorType());
  }

  bool IsTensorSequence() const noexcept {
    return (type_ != nullptr && type_->IsTensorSequenceType());
  }

  MLDataType DataType() const noexcept { return type_; }

  bool HasElement() const { return data_ != nullptr; }

 private:
  void SetType(MLDataType type) {
    ORT_ENFORCE(type != nullptr && (type->IsTensorType() || type->IsSparseTensorType()),
                "Only tensor and tensor sequences are supported can be wrapped into optional types");
    type_ = type;
  }

  std::shared_ptr<void> data_{nullptr};
  MLDataType type_{nullptr};
};

}  // namespace onnxruntime
