// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/tensor.h"

/**
   Represents both tensors and non-tensors.
*/
struct OrtValue {
 public:
  OrtValue() : data_(nullptr) {}
  virtual ~OrtValue() = default;

  OrtValue(void* pData, onnxruntime::MLDataType type, onnxruntime::DeleteFunc deleter) {
    Init(pData, type, deleter);
  }

  void Init(void* pData, onnxruntime::MLDataType type, onnxruntime::DeleteFunc deleter) {
    data_.reset(pData, deleter);
    type_ = type;
  }

  bool IsAllocated() const {
    return data_ && type_;
  }

  template <typename T>
  const T& Get() const {
    ORT_ENFORCE(onnxruntime::DataTypeImpl::GetType<T>() == type_, onnxruntime::DataTypeImpl::GetType<T>(), " != ", type_);
    return *static_cast<T*>(data_.get());
  }

  template <typename T>
  T* GetMutable() {
    ORT_ENFORCE(onnxruntime::DataTypeImpl::GetType<T>() == type_, onnxruntime::DataTypeImpl::GetType<T>(), " != ", type_);
    return static_cast<T*>(data_.get());
  }

  bool IsTensor() const noexcept {
    return (type_ != nullptr && type_->IsTensorType());
  }

  onnxruntime::MLDataType Type() const {
    return type_;
  }

  onnxruntime::Fence_t Fence() const {
    return fence_.get();
  }

  void SetFence(onnxruntime::FencePtr fence) {
    fence_ = fence;
  }

  void ShareFenceWith(OrtValue& v) {
    fence_ = v.fence_;
  }

 private:
  std::shared_ptr<void> data_;
  onnxruntime::MLDataType type_{nullptr};
  onnxruntime::FencePtr fence_;
};

template <>
inline const onnxruntime::Tensor& OrtValue::Get<onnxruntime::Tensor>() const {
  ORT_ENFORCE(IsTensor(), "Trying to get a Tensor, but got: ", onnxruntime::DataTypeImpl::ToString(type_));
  return *static_cast<onnxruntime::Tensor*>(data_.get());
}

template <>
inline onnxruntime::Tensor* OrtValue::GetMutable<onnxruntime::Tensor>() {
  ORT_ENFORCE(IsTensor(), "Trying to get a Tensor, but got: ", onnxruntime::DataTypeImpl::ToString(type_));
  return static_cast<onnxruntime::Tensor*> (data_.get());
}


//TODO: remove the following line
#define MLValue OrtValue
