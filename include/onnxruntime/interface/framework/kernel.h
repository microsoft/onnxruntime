// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "interface/framework/tensor.h"

namespace onnxruntime {

namespace interface {

struct IKernelContext {
  virtual ~IKernelContext() = default;
  virtual const void* InputData(int index) const = 0;
  virtual const int64_t* InputShape(int index, size_t* num_dims) const = 0;
  virtual void* AllocateOutput(int index, const TensorShape& shape) = 0;
};

struct IArg {
  virtual ~IArg() = default;
};

using ArgPtr = std::unique_ptr<IArg>;
using ArgPtrs = std::vector<ArgPtr>;
using TensorShape = std::vector<int64_t>;

template <typename T>
struct ITensor : public IArg {
  //using MyType = ITensor<T>;
  ITensor(IKernelContext* ctx, int index) : ctx_(ctx), index_(index){};
  const TensorShape& Shape() { return shape_; }

 protected:
  IKernelContext* ctx_ = {};
  int index_ = {};
  TensorShape shape_;
};

template <typename T>
struct TensorView : public ITensor<T> {
  TensorView(IKernelContext* ctx, int index) : ITensor<T>(ctx, index) {
    // assert ctx
    data_ = ctx->InputData(index);
    size_t num_dims = 0;
    const auto* dims = ctx->InputShape(index, num_dims);
    // assert dims
    shape_ = TensorShape{dims, dims + num_dims};
  }
  TensorView(const T* data, const TensorShape& shape) : data_(data), shape_(shape){};
  const void* Data() const {
    return data_;
  }

 protected:
  const T* data_ = {};
};

template <typename T>
struct Tensor : public ITensor<T> {
  Tensor(IKernelContext* ctx, size_t index) : ITensor<T>(ctx, index) {}
  Tensor(T* data, const TensorShape& shape) : data_(data), shape_(shape){};
  void* Allocate(const TensorShape shape) {
    if (data_) {
      return data_;
    } else {
      // assert ctx
      shape_ = shape;
      data_ = ctx->AllocateOutput(index_, shape_);
      return data_;
    }
  }

 protected:
  T* data_ = {};
};

struct IKernelInfo {
  virtual ~IKernelInfo() = default;
};

struct IKernel {
  //explicit IKernel(const IKernelInfo&){};
  //IKernel(const IKernel&) = delete;
  virtual ~IKernel() = default;
  virtual void Init(IKernelInfo&) {}; //todo - as constructor
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