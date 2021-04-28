// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/platform/threadpool.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/element_wise_ranged_transform.h"

namespace onnxruntime {

namespace functors {

template <typename T>
struct Celu : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN(alpha);

  float Cost() const final {
    // TODO: Tune the cost
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.cwiseMax(0.0f) + (((T)alpha * ((xm / (T)alpha).exp() - 1)).cwiseMin(0.0f));
  }
};

template <typename T>
struct Elu : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN(alpha);

  float Cost() const final {
    return 30.f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm >= 0).select(xm, (T)alpha * (xm.exp() - 1));
  }
};

template <typename T>
struct HardSigmoid : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN_2(alpha, beta);

  float Cost() const final {
    return 0.5f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (((T)alpha * xm + (T)beta).cwiseMin(1.0f)).cwiseMax(0.0f);
  }
};

template <typename T>
struct LeakyRelu : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN(alpha);

  float Cost() const final {
    return 25.0f;
  }

  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm >= 0).select(xm, (T)alpha * xm);
  }
};

template <typename T>
struct Softplus : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }
  float Cost() const final {
    return 15.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm > 0).select(xm + ((-xm).exp() + 1.0f).log(), ((xm).exp() + 1.0f).log());
  }
};

template <typename T>
struct Relu : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  ElementWiseRangedTransform<T>* Copy() const {  // replace it with a macro. why this?
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;  //redundant?
    return new T2(*this);
  }
  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.cwiseMax(0);
  }
};

template <typename T>
struct Sigmoid : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }
  float Cost() const final {
    return 2.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm >= 0).select(1 / (1. + (-xm.abs()).exp()), 1 - 1 / (1. + (-xm.abs()).exp()));
  }
};

template <>
void Sigmoid<float>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const;

template <typename T>
struct Softsign : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }
  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (1 + xm.abs()).inverse() * xm;
  }
};

template <typename T>
struct Tanh : public ElementWiseRangedTransform<T> {
  Status Init(const onnxruntime::NodeAttributes&) {
    return Status::OK();
  }
  ElementWiseRangedTransform<T>* Copy() const {
    using T1 = typename std::remove_pointer<decltype(this)>::type;
    using T2 = typename std::remove_const<T1>::type;
    return new T2(*this);
  }

  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = xm.tanh();
  }
};
template <>
void Tanh<float>::operator()(std::ptrdiff_t first, std::ptrdiff_t last) const;

template <typename T>
struct ThresholdedRelu : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN(alpha);

  float Cost() const final {
    return 1.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (xm > (T)alpha).select(xm, 0);
  }
};

template <typename T>
struct Selu : public ElementWiseRangedTransform<T> {
  ORT_GET_FLOAT_ATTR_AND_RETURN_2(alpha, gamma);

  float Cost() const final {
    return 4.0f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const final {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    ConstEigenVectorArrayMap<T> xm(this->input + first, len);
    EigenVectorArrayMap<T> ym(output_ptr, len);
    ym = (T)gamma * (xm.cwiseMax(0.0f) + ((T)alpha * (xm.array().exp() - 1.0f)).cwiseMin(0.0f));
  }
};

}  // namespace functors

DEFINE_ELE_KERNEL(Celu);
DEFINE_ELE_KERNEL(Elu);
DEFINE_ELE_KERNEL(HardSigmoid);
DEFINE_ELE_KERNEL(LeakyRelu);
DEFINE_ELE_KERNEL(Softplus);
DEFINE_ELE_KERNEL(Relu);
DEFINE_ELE_KERNEL(Sigmoid);
DEFINE_ELE_KERNEL(Softsign);
DEFINE_ELE_KERNEL(Tanh);
DEFINE_ELE_KERNEL(ThresholdedRelu);
DEFINE_ELE_KERNEL(Selu);

}  // namespace onnxruntime
