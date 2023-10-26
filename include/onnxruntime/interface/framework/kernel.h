// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "interface/framework/tensor.h"

namespace onnxruntime {

namespace interface {

struct IKernelContext {
  virtual ~IKernelContext() = default;
  virtual const void* InputData(int index) const = 0;
  virtual void* AllocateOutput(int index, const TensorShape& shape) = 0;
};

struct IArg {
  virtual ~IArg() = default;
};

using ArgPtr = std::unique_ptr<IArg>;
using ArgPtrs = std::vector<ArgPtr>;

template <typename T>
struct Tensor : public IArg {
  using MyType = Tensor<T>;
  Tensor(IKernelContext* ctx, size_t index, bool is_input) : ctx_(ctx), index_(index), is_input_(is_input){};
  Tensor(T* mutable_data, const TensorShape& shape) : mutable_data_(mutable_data), shape_(shape){};
  Tensor(const T* readonly_data, const TensorShape& shape) : readonly_data_(readonly_data), shape_(shape){};

  const TensorShape& Shape() const {
    if (shape_.empty()) {
      size_t num_dims = 0;
      auto dims = ctx_->InputShape(index_, &num_dims);
      if (dims) {
        for (size_t i = 0; i < num_dims; ++i) {
          shape_.push_back(dims[i]);
        }
      }  // else throw ...
    }
    return shape_;
  }
  size_t NumberOfElements() const {
    const auto& shape = Shape();
    return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>{});
  };
  const T* Data() const {
    if (!readonly_data_ && ctx_) {
      readonly_data_ = reinterpret_cast<const T*>(ctx_->InputData(index_));
    }
    return readonly_data_;
  }
  T* Allocate(const TensorShape& shape) {
    if (!mutable_data_ && ctx_) {
      mutable_data_ = reinterpret_cast<T*>(ctx_->AllocateOutput(index_, shape));
    }
    return mutable_data_;
  }

 private:
  IKernelContext* ctx_ = {};
  size_t index_ = {};
  bool is_input_ = {};
  T* mutable_data_ = {};
  mutable const T* readonly_data_ = {};
  mutable TensorShape shape_;
};

struct IKernelInfo {
  virtual ~IKernelInfo() = default;
};

struct IKernel {
  virtual ~IKernel() = default;
  virtual void Init(IKernelInfo&){};
  virtual onnxruntime::Status Compute(IKernelContext*) const = 0;

  template <size_t ith_input, size_t ith_output, typename... Ts>
  static typename std::enable_if<sizeof...(Ts) == 0, std::tuple<>>::type
  CreateTuple(IKernelContext*, ArgPtrs&) {
    return std::make_tuple();
  }

  // inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Tensor<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(IKernelContext* context, ArgPtrs& args) {
    args.push_back(std::make_unique<Tensor<float>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  /*
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const TensorT<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(IKernelContext* context, ArgPtrs& args) {
    args.push_back(std::make_unique<TensorT<float>>(context, ith_input));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const TensorV<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(IKernelContext* context, ArgPtrs& args) {
    args.push_back(std::make_unique<TensorV<float>>(context, ith_input));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  // outputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Reused<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(IKernelContext* context, ArgPtrs& args) {
    args.push_back(std::make_unique<Reused<float>>(context, ith_output));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Aliased<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(IKernelContext* context, ArgPtrs& args) {
    args.push_back(std::make_unique<Aliased<float>>(context, ith_output));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }*/
};


}  // namespace interface
}  // namespace onnxruntime