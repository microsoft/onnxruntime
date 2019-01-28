// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
/**
   Represents both tensors and non-tensors.
*/
class MLValue {
 public:
  MLValue() : data_(nullptr) {}
  virtual ~MLValue() = default;

  MLValue(void* pData, MLDataType type, DeleteFunc deleter) {
    Init(pData, type, deleter);
  }

  void Init(void* pData, MLDataType type, DeleteFunc deleter) {
    data_.reset(pData, deleter);
    type_ = type;
  }

  bool IsAllocated() const {
    return data_ && type_;
  }

  template <typename T>
  const T& Get() const {
    ORT_ENFORCE(DataTypeImpl::GetType<T>() == type_, DataTypeImpl::GetType<T>(), " != ", type_);
    return *static_cast<T*>(data_.get());
  }

  template <typename T>
  T* GetMutable() {
    ORT_ENFORCE(DataTypeImpl::GetType<T>() == type_, DataTypeImpl::GetType<T>(), " != ", type_);
    return static_cast<T*>(data_.get());
  }

  bool IsTensor() const noexcept {
    return DataTypeImpl::GetType<Tensor>() == type_;
  }

  MLDataType Type() const {
    return type_;
  }

  Fence_t Fence() const {
    return fence_.get();
  }

  void SetFence(FencePtr fence) {
    fence_ = fence;
  }

  void ShareFenceWith(MLValue& v) {
    fence_ = v.fence_;
  }

 private:
  std::shared_ptr<void> data_;
  MLDataType type_{nullptr};
  FencePtr fence_;
};
}  // namespace onnxruntime
