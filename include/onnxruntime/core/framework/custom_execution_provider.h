// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <climits>

#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/ortdevice.h"
#include "core/framework/stream_handles.h"

namespace onnxruntime {

namespace lite {

enum DataType {
  float_tp = 0,
  double_tp,
  int8_tp,
  uint8_tp,
  int16_tp,
  uint16_tp,
  int32_tp,
  uint32_tp,
  int64_tp,
  uint64_tp,
  bool_tp,
  uknownn_tp,
};

struct IKernelContext {
  virtual ~IKernelContext() = default;
};

/////////////////////////////////////////////////// Tensor Interface ///////////////////////////////////////////////////

struct ITensor {
  virtual ~ITensor() = default;
};

using TensorPtr = std::unique_ptr<ITensor>;
using TensorPtrs = std::vector<TensorPtr>;

template <typename T>
struct Tensor : public ITensor {
  using MyType = Tensor<T>;
  Tensor(IKernelContext* ctx, size_t index, bool is_input) : ctx_(ctx), index_(index), is_input_(is_input){};
  std::vector<int64_t> Shape() {
    return {};
  }
  const T* Data() const {
    return {};
  }
  T* Allocate(MyType* reuse) {
    return {};
  }

 private:
  IKernelContext* ctx_ = {};
  size_t index_;
  bool is_input_;
};

template <typename T>
struct TensorT : Tensor<T> {
  TensorT(IKernelContext* ctx, size_t index) : Tensor(ctx, index, true) {}
};

template <typename T>
struct TensorT1 : Tensor<T> {
  TensorT1(IKernelContext* ctx, size_t index) : Tensor(ctx, index, true) {}
};

template <typename T>
struct TensorV : Tensor<T> {
  TensorV(IKernelContext* ctx, size_t index) : Tensor(ctx, index, true) {}
};

template <typename T, int ith_input_to_copy_from = 0>
struct Reused : Tensor<T> {
  using MyType = Tensor<T>;
  Reused(IKernelContext* ctx, size_t index) : Tensor(ctx, index, false) {}
  T* Data() {
    return {};
  }
};

template <typename T, int ith_input_to_alias_from = 0>
struct Aliased : Reused<T, ith_input_to_alias_from> {
  using MyType = Tensor<T>;
  Aliased(IKernelContext* ctx, size_t index) : Reused(ctx, index) {}
  T* Data() {
    return {};
  }
  static int InputIndice() { return ith_input_to_alias_from; }
};

struct IKernel {
  virtual ~IKernel() = default;
  virtual onnxruntime::Status Compute(IKernelContext*) const = 0;
};

struct IKernelInfo {
  virtual ~IKernelInfo() = default;
};

template <typename... Args>
struct FnKernel : public IKernel {
  using ComputeFn = onnxruntime::Status (*)(Args...);
  FnKernel(ComputeFn compute_fn) : compute_fn_(compute_fn) {}

  template <size_t ith_input, size_t ith_output, typename... Ts>
  static typename std::enable_if<sizeof...(Ts) == 0, std::tuple<>>::type
  CreateTuple(IKernelContext*, TensorPtrs&) {
    return std::make_tuple();
  }

  // inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Tensor<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(IKernelContext* context, TensorPtrs& args) {
    args.push_back(std::make_unique<Tensor<float>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const TensorT<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(IKernelContext* context, TensorPtrs& args) {
    args.push_back(std::make_unique<TensorT<float>>(context, ith_input));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  // outputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Reused<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(IKernelContext* context, TensorPtrs& args) {
    args.push_back(std::make_unique<Reused<float>>(context, ith_output));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Aliased<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(IKernelContext* context, TensorPtrs& args) {
    args.push_back(std::make_unique<Aliased<float>>(context, ith_output));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  onnxruntime::Status Compute(IKernelContext* context) const override {
    TensorPtrs args;
    auto t = CreateTuple<0, 0, Args...>(context, args);
    return std::apply([this](Args const&... t_args) { return compute_fn_(t_args...); }, t);
  }

 private:
  ComputeFn compute_fn_;
};

/////////////////////////////////////////////////// Kernel Interface ///////////////////////////////////////////////////

struct IKernelBuilder {
  // IKernelBuilder() = default;
  // IKernelBuilder(const IKernelBuilder&) = delete;
  explicit IKernelBuilder() = default;
  IKernelBuilder(const IKernelBuilder&){};

  virtual ~IKernelBuilder() = default;
  virtual IKernelBuilder& Provider(const char*) = 0;
  virtual IKernelBuilder& SetDomain(const char*) = 0;
  virtual IKernelBuilder& SetName(const char*) = 0;
  virtual IKernelBuilder& SinceVersion(int, int) = 0;
  virtual IKernelBuilder& Alias(int, int) = 0;
  virtual IKernelBuilder& TypeConstraint(const char*, DataType) = 0;

  template <size_t, size_t, typename... Ts>
  typename std::enable_if<sizeof...(Ts) == 0, IKernelBuilder&>::type
  ParseArgs() {
    return *this;
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const TensorT<float>&>::value, IKernelBuilder&>::type
  ParseArgs() {
    return TypeConstraint("T", DataType::float_tp).ParseArgs<ith_input + 1, ith_output, Ts...>();
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const TensorT1<float>&>::value, IKernelBuilder&>::type
  ParseArgs() {
    return TypeConstraint("T1", DataType::float_tp).ParseArgs<ith_input + 1, ith_output, Ts...>();
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const TensorV<float>&>::value, IKernelBuilder&>::type
  ParseArgs() {
    return TypeConstraint("V", DataType::float_tp).ParseArgs<ith_input + 1, ith_output, Ts...>();
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, Aliased<float>&>::value, IKernelBuilder&>::type
  ParseArgs() {
    return Alias(Aliased<float>::InputIndice(), 0).ParseArgs<ith_input, ith_output + 1, Ts...>();
  }

  template <typename... Args>
  IKernelBuilder& ParseFn(onnxruntime::Status (*compute_fn)(Args...)) {
    using KernelType = FnKernel<Args...>;
    kernel_ = std::make_unique<KernelType>(compute_fn);
    return ParseArgs<0, 0, Args...>();
  }

  std::unique_ptr<IKernel> kernel_;
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
};

/////////////////////////////////////////////////// Ep Interface ///////////////////////////////////////////////////

struct IExecutionProvider {
 public:
  IExecutionProvider() { default_device_ = OrtDevice(); }
  virtual ~IExecutionProvider() = default;

  std::vector<OrtAllocator*>& GetAllocators() { return allocators_; }

  std::string& GetType() { return type_; }
  OrtDevice& GetDevice() { return default_device_; }

  virtual bool CanCopy(const OrtDevice&, const OrtDevice&) { return false; }
  virtual void MemoryCpy(Ort::UnownedValue&, Ort::ConstValue const&) {}
  virtual void RegisterStreamHandlers(IStreamCommandHandleRegistry&, std::map<OrtDevice, OrtAllocator*>&) const {};
  virtual void RegisterKernels(lite::IKernelRegistry& kernel_registry) = 0;

 protected:
  std::vector<OrtAllocator*> allocators_;
  std::string type_;
  OrtDevice default_device_;
};

}  // namespace lite

using CustomExecutionProvider = lite::IExecutionProvider;

}  // namespace onnxruntime
