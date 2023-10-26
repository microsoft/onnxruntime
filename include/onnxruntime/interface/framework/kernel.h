// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

namespace interface {

struct IKernelContext {
  virtual ~IKernelContext() = default;
  virtual const void* InputData(size_t index) const = 0;
  virtual void* AllocateOutput(size_t index, const TensorShape& shape) = 0;
};

/*
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
  }
};
*/
}  // namespace interface

}  // namespace onnxruntime