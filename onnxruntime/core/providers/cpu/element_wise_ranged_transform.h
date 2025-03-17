// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/platform/threadpool.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace functors {

inline common::Status GetFloatParam(const std::string& name, const onnxruntime::NodeAttributes& attributes,
                                    float& out) {
  auto attr = attributes.find(name);
  if (attr == attributes.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No attribute with name:'", name, "'is defined.");
  }
  if (attr->second.type() != ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOAT) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Attribute name and type don't match for '", name, "'");
  }
  out = attr->second.f();
  return common::Status::OK();
}

// Like the std::transform
// T should be float or double
// All the code(this class and its subclasses) must be exception free
template <typename T>
struct ElementWiseRangedTransform {
  using DataType = T;
  const T* input = nullptr;
  T* output = nullptr;
  // Run an unary function through the range [input + first, input + last) -> [output + first, output + last)
  // Thread safe
  virtual void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const = 0;
  // Thread safe
  virtual float Cost() const = 0;
  // Return a deep clone of *this
  virtual ElementWiseRangedTransform<T>* Copy() const = 0;
  virtual ~ElementWiseRangedTransform() = 0;

  // A helper function for creating such objects by op type name.
  // Ideally we should use op type name + domain name + op version as the key, but currently there is no conflict yet,
  // so other two are not needed
  static Status Create(const std::string& activation_type, const onnxruntime::NodeAttributes& attributes,
                       std::unique_ptr<ElementWiseRangedTransform<T>>& out);
};

template <typename T>
ElementWiseRangedTransform<T>::~ElementWiseRangedTransform() {
}
#define ORT_GET_FLOAT_ATTR_AND_RETURN(X)                           \
  float X;                                                         \
  Status Init(const onnxruntime::NodeAttributes& attributes) {     \
    return (GetFloatParam(#X, attributes, X));                     \
  }                                                                \
  GSL_SUPPRESS(r.11)                                               \
  ElementWiseRangedTransform<T>* Copy() const final {              \
    using T1 = typename std::remove_pointer<decltype(this)>::type; \
    using T2 = typename std::remove_const<T1>::type;               \
    return new T2(*this);                                          \
  }

#define ORT_GET_FLOAT_ATTR_AND_RETURN_2(X, Y)                      \
  float X;                                                         \
  float Y;                                                         \
  Status Init(const onnxruntime::NodeAttributes& attributes) {     \
    ORT_RETURN_IF_ERROR(GetFloatParam(#X, attributes, X));         \
    ORT_RETURN_IF_ERROR(GetFloatParam(#Y, attributes, Y));         \
    return Status::OK();                                           \
  }                                                                \
  GSL_SUPPRESS(r.11)                                               \
  ElementWiseRangedTransform<T>* Copy() const final {              \
    using T1 = typename std::remove_pointer<decltype(this)>::type; \
    using T2 = typename std::remove_const<T1>::type;               \
    return new T2(*this);                                          \
  }
}  // namespace functors

template <typename F>
class ElementWiseKernel final : public OpKernel {
 public:
  explicit ElementWiseKernel(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(f_.Init(info.node().GetAttributes()));
  }

  Status Compute(OpKernelContext* context) const override {
    using T = typename F::DataType;
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
    const int64_t input_size = X->Shape().Size();
    if (input_size == 0)
      return Status::OK();
    ORT_ENFORCE(input_size < std::numeric_limits<std::ptrdiff_t>::max());
    F f = f_;
    f.input = X->Data<T>();
    f.output = Y->MutableData<T>();
    concurrency::ThreadPool::TryParallelFor(tp, static_cast<std::ptrdiff_t>(input_size),
                                            {static_cast<float>(sizeof(T)), static_cast<float>(sizeof(T)), f.Cost()},
                                            f);
    return Status::OK();
  }

 private:
  F f_;
};

#define DEFINE_ELE_KERNEL(X) \
  template <typename T>      \
  using X = ElementWiseKernel<functors::X<T>>;
}  // namespace onnxruntime
