// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/ortdevice.h"
#include "core/framework/stream_handles.h"
#include "core/framework/op_kernel.h"

#include <climits>

namespace Ort {
namespace Custom {
struct ExternalKernelDef {
  std::unique_ptr<OrtLiteCustomOp> custom_op_;
  std::string domain_;
  int op_since_version_start_ = 1;
  int op_since_version_end_ = INT_MAX;
  ExternalKernelDef(OrtLiteCustomOp* op, std::string domain, int op_version_start, int op_version_end) {
    custom_op_ = std::unique_ptr<OrtLiteCustomOp>(op);
    domain_ = domain;
    op_since_version_start_ = op_version_start;
    op_since_version_end_ = op_version_end;
  }
};

template <typename... Args>
ExternalKernelDef* CreateExternalKernelDef(const char* op_name, const char* execution_provider, void (*custom_compute_fn)(Args...),
                                           const char* domain, int op_since_version_start, int op_since_version_end = INT_MAX) {
  OrtLiteCustomOp* op = CreateLiteCustomOp(op_name, execution_provider, custom_compute_fn);
  return std::make_unique<ExternalKernelDef>(op, domain, op_since_version_start, op_since_version_end).release();
}

}  // namespace Custom
}  // namespace Ort

namespace onnxruntime {

////////////////////////////////////////////////// lite tensors //////////////////////////////////////////////////
namespace lite {

struct Arg {};

using ArgPtr = std::unique_ptr<Arg>;
using ArgPtrs = std::vector<ArgPtr>;

template <typename T>
struct Tensor : public Arg {
  using MyType = Tensor<T>;
  Tensor(onnxruntime::OpKernelContext* ctx, size_t index, bool is_input) : ctx_(ctx), index_(index), is_input_(is_input){};
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
  onnxruntime::OpKernelContext* ctx_ = {};
  size_t index_;
  bool is_input_;
};

template <typename T, int64_t ith_input_to_copy_from = 0>
struct Reused : Tensor<T> {
  using MyType = Tensor<T>;
  Reused(onnxruntime::OpKernelContext* ctx, size_t index) : Tensor(ctx, index, false) {}
  T* Data() {
    return {};
  }
};

template <typename T, int64_t ith_input_to_alias_from = 0>
struct Aliasd : Reused<T, ith_input_to_alias_from> {
  using MyType = Tensor<T>;
  Aliasd(onnxruntime::OpKernelContext* ctx, size_t index) : Reused(ctx, index) {}
  T* Data() {
    return {};
  }
};

// onnxruntime::Status Conv(const Tensor<float>& /*X*/,
//                          const Tensor<float>& /*W*/,
//                          const Tensor<float>& /*B*/,
//                          Tensor<float>& /*Y*/);
//
// onnxruntime::Status Relu(const Tensor<float>& /*X*/, Reused<float>& /*Y*/);
//
// onnxruntime::Status Identity(const Tensor<float>& /*X*/, Aliasd<float>& /*Y*/);

struct Kernel {};

template <typename... Args>
struct LiteKernelFn : public onnxruntime::OpKernel {
  using ComputeFn = onnxruntime::Status (*)(Args...);
  LiteKernelFn(const OpKernelInfo& info, ComputeFn compute_fn) : OpKernel(info), compute_fn_(compute_fn) {}

  template <size_t ith_input, size_t ith_output, typename... Ts>
  static typename std::enable_if<sizeof...(Ts) == 0, std::tuple<>>::type
  CreateTuple(OrtKernelContext*, ArgPtrs&) {
    return std::make_tuple();
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Tensor<float>&>::value, std::tuple <T, Ts...>> ::type
  CreateTuple(onnxruntime::OpKernelContext* context, ArgPtrs& args) {
    args.push_back(std::make_unique<Tensor<float>>(context, ith_input, True));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Reused<float>&>::value, std::tuple <T, Ts...>> ::type
  CreateTuple(onnxruntime::OpKernelContext* context, ArgPtrs& args) {
    args.push_back(std::make_unique<Reused<float>>(context, ith_output, True));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*args.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(context, args);
    return std::tuple_cat(current, next);
  }

  onnxruntime::Status Compute(onnxruntime::OpKernelContext* context) const override {
    ArgPtrs args;
    auto t = CreateTuple<0, 0, Args...>(context, args);
    return std::apply([this](Args const&... t_args) { compute_fn_(t_args...); }, t);
  }

 private:
  ComputeFn compute_fn_;
};

template <typename... Args>
onnxruntime::KernelDefBuilder& RegisterKernel(onnxruntime::Status (*)(Args...)) {
  onnxruntime::KernelCreateInfo;
  return {};
}

}  // namespace lite
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CustomExecutionProvider {
 public:
  CustomExecutionProvider() { default_device_ = OrtDevice(); };
  virtual ~CustomExecutionProvider() = default;

  std::vector<OrtAllocator*>& GetAllocators() { return allocators_; }
  // std::vector<std::unique_ptr<Ort::Custom::OrtLiteCustomOp>>& GetKernelDefinitions() { return kernel_definitions_; }
  size_t GetKernelDefinitionCount() { return kernel_definitions_.size(); }
  Ort::Custom::ExternalKernelDef* GetKernelDefinition(size_t index) {
    if (index >= kernel_definitions_.size()) return nullptr;
    return kernel_definitions_[index].get();
  }
  std::string& GetType() { return type_; }
  OrtDevice& GetDevice() { return default_device_; }

  virtual bool CanCopy(const OrtDevice&, const OrtDevice&) { return false; }
  // virtual void MemoryCpy(OrtValue&, const OrtValue&) {}
  virtual void MemoryCpy(Ort::UnownedValue&, Ort::ConstValue const&) {}
  virtual void RegisterStreamHandlers(IStreamCommandHandleRegistry&, std::map<OrtDevice, OrtAllocator*>&) const {}

/////////////////////////////////////////////////// unified kenrel registration ///////////////////////////////////////////////////

  //throw on err
  template<typename... Args>
  void RegisterKernel(const char* name,
                      onnxruntime::Status (*)(Args... args),
                      size_t start_ver = 0,
                      size_t end_ver = (1<<30)){};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 protected:
  std::vector<OrtAllocator*> allocators_;
  // std::vector<std::unique_ptr<Ort::Custom::OrtLiteCustomOp>> kernel_definitions_;
  std::vector<std::unique_ptr<Ort::Custom::ExternalKernelDef>> kernel_definitions_;
  std::string type_;
  OrtDevice default_device_;
};

}  // namespace onnxruntime
