// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_kernel.h"

#define II(i) (static_cast<int>(i))

namespace onnxruntime {

namespace lite {

struct Arg {
  Arg(onnxruntime::OpKernelContext* ctx,
      size_t indice, bool is_input) : ctx_(ctx), indice_(indice), is_input_(is_input) {}
  virtual ~Arg(){};

 protected:
  onnxruntime::OpKernelContext* ctx_;
  size_t indice_;
  bool is_input_;
};

template <typename T>
struct Tensor : public Arg {
  Tensor(OpKernelContext* ctx,
         size_t indice, bool is_input) : Arg(ctx, indice, is_input) {}
  const T* Data() const {
    ORT_ENFORCE(is_input_);
    const onnxruntime::Tensor* tensor = ctx_->Input<onnxruntime::Tensor>(II(indice_));
    return tensor->Data<T>();
  }
  std::vector<int64_t> Shape() const {
    ORT_ENFORCE(is_input_);
    const onnxruntime::Tensor* tensor = ctx_->Input<onnxruntime::Tensor>(II(indice_));
    std::vector<int64_t> shape;
    for (auto dim : tensor->Shape().AsShapeVector()) {
      shape.push_back(dim);
    }
    return std::move(shape);
  }
  int64_t NumberOfElement() const {
    ORT_ENFORCE(is_input_);
    const onnxruntime::Tensor* tensor = ctx_->Input<onnxruntime::Tensor>(II(indice_));
    return tensor->Shape().Size();
  }
  T* Allocate(const std::vector<int64_t>& shape) {
    ORT_ENFORCE(!is_input_);
    onnxruntime::Tensor* tensor = ctx_->Output(II(indice_), shape);
    return tensor->MutableData<T>();
  }

 protected:
  std::vector<int64_t> shape_;
};

using ArgPtr = std::unique_ptr<lite::Arg>;
using ArgPtrs = std::vector<ArgPtr>;

template <typename... Args>
struct OpLite : public onnxruntime::OpKernel {
  using ComputeFn = onnxruntime::Status (*)(Args...);

  // constructor
  OpLite(const OpKernelInfo& info, ComputeFn compute_fn) : OpKernel(info) {
    compute_fn_ = compute_fn;
  }

  // end of tuple
  template <size_t ith_input, size_t ith_output, typename... Ts>
  static typename std::enable_if<sizeof...(Ts) == 0, std::tuple<>>::type
  CreateTuple(OpKernelContext*, ArgPtrs&) {
    return std::make_tuple();
  }

  // inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Tensor<int>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(OpKernelContext* context, ArgPtrs& args) {
    args.push_back(std::make_unique<Tensor<int>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Tensor<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(OpKernelContext* context, ArgPtrs& args) {
    args.push_back(std::make_unique<Tensor<float>>(context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  // outputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Tensor<int>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(OpKernelContext* context, ArgPtrs& args) {
    args.push_back(std::make_unique<Tensor<int>>(context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Tensor<float>&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(OpKernelContext* context, ArgPtrs& args) {
    args.push_back(std::make_unique<Tensor<float>>(context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  // compute
  Status Compute(_Inout_ onnxruntime::OpKernelContext* context) const override {
    ArgPtrs args;
    auto t = CreateTuple<0, 0, Args...>(context, args);
    return std::apply([this](Args const&... t_args) { return this->compute_fn_(t_args...); }, t);
  }

 private:
  ComputeFn compute_fn_;
};

template <typename... Args>
onnxruntime::OpKernel* CreateLiteOp(const OpKernelInfo& info, onnxruntime::Status (*compute_fn)(Args...)) {
  using Op = OpLite<Args...>;
  return std::make_unique<Op>(info, compute_fn).release();
}

#define ONNX_OP_BY_FN(name, domain, ver, provider, builder, fn)                           \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name);                     \
  template <>                                                                             \
  KernelCreateInfo                                                                        \
  BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>() { \
    return KernelCreateInfo(                                                              \
        builder.SetName(#name)                                                            \
            .SetDomain(domain)                                                            \
            .SinceVersion(ver)                                                            \
            .Provider(provider)                                                           \
            .Build(),                                                                     \
        static_cast<KernelCreatePtrFn>(                                                   \
            [](FuncManager&,                                                              \
               const OpKernelInfo& info,                                                  \
               std::unique_ptr<OpKernel>& out) -> Status {                                \
              out.reset(CreateLiteOp(info, fn));                                          \
              return Status::OK();                                                        \
            }));                                                                          \
  }

}  // namespace lite

}  // namespace onnxruntime