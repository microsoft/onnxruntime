// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <numeric>
#include <vector>
#include <memory>
#include <functional>
#include <interface/common/data_types.h>

namespace onnxruntime {

namespace interface {
struct ITensorShape {
  virtual int64_t NumberOfElements() const = 0;
  virtual const int64_t* GetDimensions(size_t& num_dims) const = 0;
};

struct IArg {
  virtual ~IArg() = default;
};

using IArgPtr = std::unique_ptr<IArg>;
using IArgPtrs = std::vector<IArgPtr>;

struct ITensor : public IArg {
  virtual const ITensorShape& GetShape() const = 0;
  virtual const void* GetRawData() const {
    return {};
  }
  virtual void* GetMutableRawData() {
    return {};
  }
  virtual TensorDataType GetDataType() const = 0;
};

// readonly tensors
template <typename T>
struct IReadonlyTensor : public ITensor {
  virtual const T* GetData() const = 0;
};

template <typename T>
struct ReadonlyTensor : public IReadonlyTensor<T> {
  using DataType = const T*;
  ReadonlyTensor(const ITensor& readonly_tensor) : readonly_tensor_(readonly_tensor) {}
  const ITensorShape& GetShape() const override { return readonly_tensor_.GetShape(); }
  DataType GetData() const override { return reinterpret_cast<DataType>(readonly_tensor_.GetRawData()); }
  TensorDataType GetDataType() const override { return readonly_tensor_.GetDataType(); }
  T AsScalar() const { return GetData()[0]; }
  const ITensor& readonly_tensor_;
};

// mutable tensors
template <typename T>
struct IMutableTensor : public ITensor {
  virtual T* Allocate(const ITensorShape& shape) = 0;
};

template <typename T>
struct MutableTensor : public IMutableTensor<T> {
  using AllocFn = std::function<ITensor*(const int64_t*, size_t)>;
  MutableTensor(AllocFn alloc_fn) : alloc_fn_(alloc_fn) {}
  const ITensorShape& GetShape() const override {
    // assert mutable_tensor_
    return mutable_tensor_->GetShape();
  }
  T* Allocate(const ITensorShape& shape) override {
    if (!mutable_tensor_) {
      size_t num_dims = 0;
      const int64_t* dims = shape.GetDimensions(num_dims);
      mutable_tensor_ = alloc_fn_(dims, num_dims);
    }
    return reinterpret_cast<T*>(mutable_tensor_->GetMutableRawData());
  }
  TensorDataType GetDataType() const override {
    // assert mutable_tensor_
    return mutable_tensor_->GetDataType();
  }
  AllocFn alloc_fn_;
  ITensor* mutable_tensor_ = {};
};

struct ITensorSeq : public IArg {
  virtual size_t Size() const = 0;
  virtual const ITensor& GetTensor(int64_t indice) const = 0;
  virtual void InsertTensor(ITensor& /*tensor*/, int64_t /*indice = -1*/){};
  virtual void CloneFrom(const ITensorSeq&){};
};

template <typename T>
struct IReadonlyTensorSeq : public ITensorSeq {
  virtual const IReadonlyTensor<T>& GetReadOnly(int64_t indice) const = 0;
};

template <typename T>
struct ReadonlyTensorSeq : public IReadonlyTensorSeq<T> {
  ReadonlyTensorSeq(const ITensorSeq& tensor_seq) : tensor_seq_(tensor_seq) {}
  size_t Size() const override { return tensor_seq_.Size(); }
  const ITensor& GetTensor(int64_t indice) const override { return tensor_seq_.GetTensor(indice); }
  const IReadonlyTensor<T>& GetReadOnly(int64_t indice) const override {
    return reinterpret_cast<const IReadonlyTensor<T>&>(GetTensor(indice));
  }
  const ITensorSeq& tensor_seq_;
};

template <typename T>
struct IMutableTensorSeq : public ITensorSeq {
  virtual void CloneFromReadonly(const IReadonlyTensorSeq<T>& tensor_seq) = 0;
};

template <typename T>
struct MutableTensorSeq : public IMutableTensorSeq<T> {
  MutableTensorSeq(ITensorSeq& tensor_seq) : tensor_seq_(tensor_seq) {}
  void CloneFromReadonly(const IReadonlyTensorSeq<T>& tensor_seq) override {
    tensor_seq_.CloneFrom(tensor_seq);
  }
  size_t Size() const override { return tensor_seq_.Size(); }
  const ITensor& GetTensor(int64_t indice) const override { return tensor_seq_.GetTensor(indice); }
  void InsertTensor(ITensor& tensor, int64_t indice) override {
    tensor_seq_.InsertTensor(tensor, indice);
  }
  ITensorSeq& tensor_seq_;
};

}  // namespace interface

}  // namespace onnxruntime