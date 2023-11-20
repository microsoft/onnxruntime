// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include <numeric>
#include "interface/framework/tensor.h"
#include "core/common/status.h"

namespace onnxruntime {

namespace interface {

struct IKernelContext {
  virtual ~IKernelContext() = default;
  virtual const ITensor& GetInputTensor(int index) const = 0;
  virtual const ITensorShape& GetInputShape(int index) const = 0;
  virtual ITensor* AllocOutputTensor(int index, const int64_t* dims, size_t num_dims) = 0;
};

struct IKernelInfo {
  virtual ~IKernelInfo() = default;
};

struct IKernel {
  explicit IKernel() = default;
  virtual ~IKernel() = default;
  virtual onnxruntime::Status Compute(IKernelContext*) const = 0;

  template <int ith_input, int ith_output, typename... Ts>
  static typename std::enable_if<sizeof...(Ts) == 0, std::tuple<>>::type
  CreateTuple(IKernelContext*, IArgPtrs&) {
    return std::make_tuple();
  }

  // inputs
  template <int ith_input, int ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, IReadonlyTensor<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(IKernelContext* context, IArgPtrs& args) {
    const ITensor& input_tensor = context->GetInputTensor(ith_input);
    args.push_back(std::make_unique<ReadonlyTensor<float>>(input_tensor));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  // outputs
  template <int ith_input, int ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, IMutableTensor<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(IKernelContext* context, IArgPtrs& args) {
    MutableTensor<float>::AllocFn alloc_fn = [context](const int64_t* dims, size_t num_dims) {
      return context->AllocOutputTensor(ith_output, dims, num_dims);
    };
    args.push_back(std::make_unique<MutableTensor<float>>(alloc_fn));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }
};

template <typename... Args>
struct FnKernel : public IKernel {
  using ComputeFn = onnxruntime::Status (*)(Args...);
  FnKernel(ComputeFn compute_fn) : compute_fn_(compute_fn) {}

  onnxruntime::Status Compute(IKernelContext* context) const override {
    IArgPtrs args;
    auto t = CreateTuple<0, 0, Args...>(context, args);
    return std::apply([this](Args const&... t_args) { return compute_fn_(t_args...); }, t);
  }

 private:
  ComputeFn compute_fn_;
};

template <typename K>
struct StructKernel : public IKernel {
  template <typename... Args>
  using ComputeFn = onnxruntime::Status (K::*)(Args...);

  explicit StructKernel(const IKernelInfo& info) {
    kernel_ = std::make_unique<K>(info);
  }
  onnxruntime::Status Compute(IKernelContext* context) const override {
    return InvokeCompute(&K::Compute, context);
  }

  template <typename... Args>
  onnxruntime::Status InvokeCompute(ComputeFn<Args...>, IKernelContext* context) const {
    IArgPtrs args;
    auto t = CreateTuple<0, 0, Args...>(context, args);
    return std::apply([this](Args const&... t_args) { return kernel_->Compute(t_args...); }, t);
  }
  std::unique_ptr<K> kernel_;
};

struct IKernelBuilder {
  explicit IKernelBuilder() = default;
  IKernelBuilder(const IKernelBuilder&) = delete;
  virtual ~IKernelBuilder() = default;
  virtual IKernelBuilder& Provider(const char*) = 0;
  virtual IKernelBuilder& SetDomain(const char*) = 0;
  virtual IKernelBuilder& SetName(const char*) = 0;
  virtual IKernelBuilder& SinceVersion(int, int) = 0;
  virtual IKernelBuilder& Alias(int, int) = 0;
  virtual IKernelBuilder& TypeConstraint(const char*, TensorDataType) = 0;

  template <size_t, size_t, typename... Ts>
  typename std::enable_if<sizeof...(Ts) >= 0, IKernelBuilder&>::type
  ParseArgs() {
    // todo - generate constraints by args...
    return *this;
  }

  template <typename... Args>
  IKernelBuilder& ParseFn(onnxruntime::Status (*compute_fn)(Args...)) {
    using KernelType = FnKernel<Args...>;
    create_kernel_fn_ = [compute_fn](const IKernelInfo&) {
      return std::make_unique<KernelType>(compute_fn);
    };
    return ParseArgs<0, 0, Args...>();
  }

  template <typename K, typename... Args>
  IKernelBuilder& ParseStruct(onnxruntime::Status (K::*)(Args...)) {
    using KernelType = StructKernel<K>;
    create_kernel_fn_ = [&](const IKernelInfo& info) {
      return std::make_unique<KernelType>(info);
    };
    return ParseArgs<0, 0, Args...>();
  }

  std::function<std::unique_ptr<IKernel>(const IKernelInfo&)> create_kernel_fn_ =
      [](const IKernelInfo&) -> std::unique_ptr<IKernel> { return {}; };
};

struct IKernelRegistry {
  virtual ~IKernelRegistry() = default;
  virtual IKernelBuilder& CreateBuilder() = 0;
  template <typename... Args>
  IKernelBuilder& RegisterKernel(const char* ep,
                                 const char* domain,
                                 const char* op,
                                 int since_ver,
                                 int end_ver,
                                 onnxruntime::Status (*compute_fn)(Args...)) {
    return CreateBuilder().Provider(ep).SetDomain(domain).SetName(op).SinceVersion(since_ver, end_ver).ParseFn<Args...>(compute_fn);
  }
  template <typename K, typename... Args>
  IKernelBuilder& RegisterKernel(const char* ep,
                                 const char* domain,
                                 const char* op,
                                 int since_ver,
                                 int end_ver) {
    return CreateBuilder().Provider(ep).SetDomain(domain).SetName(op).SinceVersion(since_ver, end_ver).ParseStruct<K, Args...>(&K::Compute);
  }
};

}  // namespace interface
}  // namespace onnxruntime
