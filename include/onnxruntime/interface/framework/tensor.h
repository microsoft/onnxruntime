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

template <typename T>
struct MutableTensorRef : public IMutableTensor<T> {
  MutableTensorRef(IMutableTensor<T>& ref) : ref_(ref) {}
  const ITensorShape& GetShape() const override {
    return ref_.GetShape();
  }
  T* Allocate(const ITensorShape& shape) override {
    return ref_.Allocate(shape);
  }
  TensorDataType GetDataType() const override {
    return ref_.GetDataType();
  }
  IMutableTensor<T>& ref_;
};

// struct ITensorSeq

}  // namespace interface

}  // namespace onnxruntime