// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include <vector>
#include <utility>

namespace onnxruntime {
// Put this in a separate file to avoid circular dependency between tensor.h and data_types.h
// Data type to represent a sequence of tensors of the same type
class TensorSeq {
 public:
  TensorSeq() = default;
  explicit TensorSeq(MLDataType elem_type) noexcept {
    SetType(elem_type);
  }

  using const_iterator = std::vector<Tensor>::const_iterator;

  // Sets the element type after construction.
  // Expects sequence to be empty at the time.
  void SetType(MLDataType elem_type) {
    assert(tensors_.empty());
    elem_type_ = elem_type->AsPrimitiveDataType();
    ORT_ENFORCE(elem_type_ != nullptr, "Tensor sequence must contain only primitive types");
  }

  void SetElements(std::vector<Tensor>&& tensors) {
    // The caller of this method ensures that :
    // (1) `elem_type` is set before invoking this method
    // (2) All tensors contain elements of the same primitive data type
    assert(tensors_.empty());
    tensors_ = std::move(tensors);
  }

  MLDataType DataType() const noexcept { return elem_type_; }

  bool IsSameDataType(const TensorSeq& o) const noexcept {
    return elem_type_ == o.elem_type_;
  }

  bool IsSameDataType(const Tensor& o) const noexcept {
    return elem_type_ == o.DataType()->AsPrimitiveDataType();
  }

  size_t Size() const noexcept { return tensors_.size(); }

  // Suitable for for range loop
  const_iterator begin() const noexcept {
    return tensors_.cbegin();
  }

  const_iterator end() const noexcept {
    return tensors_.cend();
  }

  // Get by index
  const Tensor& Get(size_t i) const {
    ORT_ENFORCE(i < tensors_.size());
    return tensors_[i];
  }

  void Add(Tensor&& tensor) {
    tensors_.push_back(std::move(tensor));
  }

 private:
  // A sequence must be associated with only one data type and all tensors in the seq must be of that type
  // One other alternative of storing the data type of a seq is to templatize the TensorSeq class.
  // The current design follows the Tensor methodology.
  // We also require this because the SequenceEmpty op expects the creation of a seq of a specific type
  // and the SequenceInsert op expects validation of tensors to be added to the seq against this type.
  const PrimitiveDataTypeBase* elem_type_{};

  // TODO: optimization opportunity - if all tensors in the seq are scalars, we can potentially represent them
  // as vector<primitive type>
  std::vector<Tensor> tensors_;
};

}  // namespace onnxruntime
