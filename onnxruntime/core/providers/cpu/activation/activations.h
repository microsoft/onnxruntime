// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

#define EIGEN_X ConstEigenVectorArrayMap<T>(X->template Data<T>(), X->Shape().Size())
#define EIGEN_X_VAR(var) ConstEigenVectorArrayMap<T> var(X->template Data<T>(), X->Shape().Size())
#define EIGEN_Y EigenVectorArrayMap<T>(Y->template MutableData<T>(), Y->Shape().Size())
#define EIGEN_Y_VAR(var) EigenVectorArrayMap<T> var(Y->template MutableData<T>(), Y->Shape().Size())

template <typename T>
class Elu final : public OpKernel {
 public:
  Elu(const OpKernelInfo& info) : OpKernel(info), alpha_(info.GetAttrOrDefault("alpha", 1.0f)) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (xm >= 0).select(xm, (T)alpha_ * (xm.exp() - 1));
    return Status::OK();
  }

 private:
  const float alpha_;
};

template <typename T>
class HardSigmoid final : public OpKernel {
 public:
  HardSigmoid(const OpKernelInfo& info)
      : OpKernel(info), alpha_(info.GetAttrOrDefault("alpha", 0.2f)), beta_(info.GetAttrOrDefault("beta", 0.5f)) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y_VAR(ym);
    ym = (((T)alpha_ * xm + (T)beta_).cwiseMin(1.0f)).cwiseMax(0.0f);
    return Status::OK();
  }

 private:
  const float alpha_;
  const float beta_;
};

template <typename T>
class LeakyRelu final : public OpKernel {
 public:
  LeakyRelu(const OpKernelInfo& info) : OpKernel(info), alpha_(info.GetAttrOrDefault("alpha", 0.01f)) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (xm >= 0).select(xm, (T)alpha_ * xm);
    return Status::OK();
  }

 private:
  const float alpha_;
};

template <typename T>
class ParametricSoftplus final : public OpKernel {
 public:
  ParametricSoftplus(const OpKernelInfo& info)
      : OpKernel(info), alpha_(info.GetAttrOrDefault("alpha", 1.0f)), beta_(info.GetAttrOrDefault("beta", 1.0f)) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (T)alpha_ *
              (xm * (T)beta_ > 0)
                  .select(xm * (T)beta_ + ((-xm * (T)beta_).exp() + 1.0f).log(), ((xm * (T)beta_).exp() + 1.0f).log());
    return Status::OK();
  }

 private:
  const float alpha_;
  const float beta_;
};

template <typename T>
class Relu : public OpKernel {
 public:
  Relu(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_Y = EIGEN_X.cwiseMax(0);
    return Status::OK();
  }
};

template <typename T>
class Selu final : public OpKernel {
 public:
  // TODO: I don't think float can represent such a long string(1.67326319217681884765625)
  Selu(const OpKernelInfo& info)
      : OpKernel(info),
        alpha_(info.GetAttrOrDefault("alpha", 1.67326319217681884765625f)),
        gamma_(info.GetAttrOrDefault("gamma", 1.05070102214813232421875f)) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (T)gamma_ * (xm.cwiseMax(0.0f) + ((T)alpha_ * (xm.array().exp() - 1.0f)).cwiseMin(0.0f));
    return Status::OK();
  }

 private:
  const float alpha_;
  const float gamma_;
};

template <typename T>
class Sigmoid final : public OpKernel {
 public:
  Sigmoid(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y_VAR(ym);
    ym = (xm >= 0).select(1 / (1. + (-xm.abs()).exp()), 1 - 1 / (1. + (-xm.abs()).exp()));
    return Status::OK();
  }
};

template <>
Status Sigmoid<float>::Compute(OpKernelContext* context) const;

template <typename T>
class Softsign final : public OpKernel {
 public:
  Softsign(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (1 + xm.abs()).inverse() * xm;
    return Status::OK();
  }
};

template <typename T>
class Tanh final : public OpKernel {
 public:
  Tanh(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_Y = EIGEN_X.tanh();
    return Status::OK();
  }
};

template <>
Status Tanh<float>::Compute(OpKernelContext* context) const;

template <typename T>
class ThresholdedRelu final : public OpKernel {
 public:
  ThresholdedRelu(const OpKernelInfo& info) : OpKernel(info), alpha_(info.GetAttrOrDefault("alpha", 1.0f)) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    EIGEN_X_VAR(xm);
    EIGEN_Y = (xm > (T)alpha_).select(xm, 0);
    return Status::OK();
  }

 private:
  const float alpha_;
};
}  // namespace onnxruntime
