// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/data_types.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "interface/framework/tensor.h"
#include <vector>
#include <utility>

namespace onnxruntime {
// Put this in a separate file to avoid circular dependency between tensor.h and data_types.h
// Data type to represent a sequence of tensors of the same type
class TensorSeq : public interface::ITensorSeq {
 public:
  TensorSeq() = default;
  explicit TensorSeq(MLDataType elem_type) noexcept {
    SetType(elem_type);
  }

  using const_iterator = std::vector<OrtValue>::const_iterator;
  using iterator = std::vector<OrtValue>::iterator;

  // Sets the element type after construction.
  // Expects sequence to be empty at the time.
  void SetType(MLDataType elem_type) {
    assert(tensors_.empty());
    elem_type_ = elem_type->AsPrimitiveDataType();
    ORT_ENFORCE(elem_type_ != nullptr, "Tensor sequence must contain only primitive types");
  }

  void SetElements(std::vector<OrtValue>&& tensors) {
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

  size_t Size() const override { return tensors_.size(); }

  // Suitable for for range loop
  const_iterator begin() const noexcept {
    return tensors_.cbegin();
  }

  const_iterator end() const noexcept {
    return tensors_.cend();
  }

  iterator begin() noexcept {
    return tensors_.begin();
  }

  iterator end() noexcept {
    return tensors_.end();
  }

  // Get onnxruntime::Tensor by index
  const Tensor& Get(size_t i) const {
    return GetAt(i).Get<Tensor>();
  }

  // Get OrtValue by index
  const OrtValue& GetAt(size_t i) const {
    ORT_ENFORCE(i < tensors_.size());
    return tensors_[i];
  }

  const interface::ITensor& GetTensor(int64_t indice) const override {
    return Get(static_cast<size_t>(indice));
  }

  void CloneFrom(const ITensorSeq& tensor_seq) override {
    auto source_tensor_seq = reinterpret_cast<const TensorSeq&>(tensor_seq);
    for (const auto& tensor : source_tensor_seq.tensors_) {
      Add(tensor);
    }
  }

  void Add(const OrtValue& tensor) {
    ORT_ENFORCE(IsSameDataType(tensor.Get<Tensor>()),
                "TensorSeq: tensor to be added has a different data type.");
    tensors_.push_back(tensor);
  }

  void Add(OrtValue&& tensor) {
    ORT_ENFORCE(IsSameDataType(tensor.Get<Tensor>()),
                "TensorSeq: tensor to be added has a different data type.");
    tensors_.push_back(std::move(tensor));
  }

  void Add(Tensor&& tensor) {
    ORT_ENFORCE(IsSameDataType(tensor),
                "TensorSeq: tensor to be added has a different data type.");
    OrtValue value;
    Tensor::InitOrtValue(std::move(tensor), value);
    Add(std::move(value));
  }

  void InsertTensor(interface::ITensor& tensor, int64_t indice) override {
    Tensor& source_tensor = reinterpret_cast<Tensor&>(tensor);
    OrtValue value(&source_tensor, TensorTypeBase::Type(), [](void*) {});
    tensors_.insert(tensors_.begin() + indice, value);
  }

  static void InitOrtValue(const TensorSeq& source_tensor_seq, std::shared_ptr<IAllocator> allocator, OrtValue& ort_value) {
    auto target_tensor_seq = std::make_unique<TensorSeq>(source_tensor_seq.DataType());
    target_tensor_seq->Reserve(source_tensor_seq.Size());
    for (auto iter = source_tensor_seq.begin(); iter != source_tensor_seq.end(); ++iter) {
      const Tensor& tensor = iter->Get<Tensor>();
      OrtValue value;
      Tensor::InitOrtValue(tensor.DataType(), tensor.Shape(), allocator, value);
      target_tensor_seq->Add(std::move(value));
    }

    auto ml_tensor_seq = SequenceTensorTypeBase::Type();
    ort_value.Init(target_tensor_seq.release(), ml_tensor_seq, ml_tensor_seq->GetDeleteFunc());
  }

  void Reserve(size_t capacity) {
    tensors_.reserve(capacity);
  }

 private:
  // A sequence must be associated with only one data type and all tensors in the seq must be of that type
  // One other alternative of storing the data type of a seq is to templatize the TensorSeq class.
  // The current design follows the Tensor methodology.
  // We also require this because the SequenceEmpty op expects the creation of a seq of a specific type
  // and the SequenceInsert op expects validation of tensors to be added to the seq against this type.
  const PrimitiveDataTypeBase* elem_type_{};

  std::vector<OrtValue> tensors_;
};

}  // namespace onnxruntime
