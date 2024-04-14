// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/unary_elementwise_ops.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/activation/activations.h"
#include "activations_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class Affine final : public UnaryElementwise {
 public:
  Affine(const OpKernelInfo& info) : UnaryElementwise(info) {
    ORT_ENFORCE(info.GetAttr("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr("beta", &beta_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_ALPHA_BETA()

  float alpha_;
  float beta_;
};

template <typename T>
class ParametricSoftplus final : public UnaryElementwise {
 public:
  ParametricSoftplus(const OpKernelInfo& info) : UnaryElementwise(info) {
    ORT_ENFORCE(info.GetAttr("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr("beta", &beta_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_ALPHA_BETA()

  float alpha_;
  float beta_;
};

template <typename T>
class ScaledTanh final : public UnaryElementwise {
 public:
  ScaledTanh(const OpKernelInfo& info) : UnaryElementwise(info) {
    ORT_ENFORCE(info.GetAttr("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr("beta", &beta_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_ALPHA_BETA()

  float alpha_;
  float beta_;
};

template <typename T>
class QuickGelu final : public UnaryElementwise {
 public:
  QuickGelu(const OpKernelInfo& info) : UnaryElementwise(info) {
    alpha_ = info.GetAttrOrDefault<float>("alpha", 1.702f);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_ALPHA()
  float alpha_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
