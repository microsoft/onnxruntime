// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/unary_elementwise_ops.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "activations_impl.h"

namespace onnxruntime {
namespace cuda {

#define MAKE_FUNC_CTX_ALPHA()           \
  inline CtxAlpha MakeFuncCtx() const { \
    CtxAlpha ctx;                       \
    ctx.alpha = alpha_;                 \
    return ctx;                         \
  }

#define MAKE_FUNC_CTX_ALPHA_BETA()          \
  inline CtxAlphaBeta MakeFuncCtx() const { \
    CtxAlphaBeta ctx;                       \
    ctx.alpha = alpha_;                     \
    ctx.beta = beta_;                       \
    return ctx;                             \
  }

#define MAKE_FUNC_CTX_ALPHA_GAMMA()          \
  inline CtxAlphaGamma MakeFuncCtx() const { \
    CtxAlphaGamma ctx;                       \
    ctx.alpha = alpha_;                      \
    ctx.gamma = gamma_;                      \
    return ctx;                              \
  }

#define MAKE_FUNC_CTX_NULL()           \
  inline CtxNull MakeFuncCtx() const { \
    CtxNull ctx;                       \
    return ctx;                        \
  }

template <typename T>
class Elu final : public UnaryElementwise {
 public:
  Elu(const OpKernelInfo& info) : UnaryElementwise(info) {
    ORT_ENFORCE(info.GetAttr("alpha", &alpha_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_ALPHA()

  float alpha_;
};

template <typename T>
class HardSigmoid final : public UnaryElementwise {
 public:
  HardSigmoid(const OpKernelInfo& info) : UnaryElementwise(info) {
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
class LeakyRelu final : public UnaryElementwise {
 public:
  LeakyRelu(const OpKernelInfo& info) : UnaryElementwise(info) {
    ORT_ENFORCE(info.GetAttr("alpha", &alpha_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_ALPHA()

  float alpha_;
};

template <typename T>
class Relu final : public UnaryElementwise {
 public:
  Relu(const OpKernelInfo& info) : UnaryElementwise(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_NULL()
};

template <typename T>
class Selu final : public UnaryElementwise {
 public:
  Selu(const OpKernelInfo& info) : UnaryElementwise(info) {
    ORT_ENFORCE(info.GetAttr("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr("gamma", &gamma_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_ALPHA_GAMMA()

  float alpha_;
  float gamma_;
};

template <typename T>
class Sigmoid final : public UnaryElementwise {
 public:
  Sigmoid(const OpKernelInfo& info) : UnaryElementwise(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_NULL()
};

template <typename T>
class Softplus final : public UnaryElementwise {
 public:
  Softplus(const OpKernelInfo& info) : UnaryElementwise(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_NULL()
};

template <typename T>
class Softsign final : public UnaryElementwise {
 public:
  Softsign(const OpKernelInfo& info) : UnaryElementwise(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_NULL()
};

template <typename T>
class Tanh final : public UnaryElementwise {
 public:
  Tanh(const OpKernelInfo& info) : UnaryElementwise(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_NULL()
};

template <typename T>
class ThresholdedRelu final : public UnaryElementwise {
 public:
  ThresholdedRelu(const OpKernelInfo& info) : UnaryElementwise(info) {
    ORT_ENFORCE(info.GetAttr("alpha", &alpha_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_ALPHA()
  float alpha_;
};

}  // namespace cuda
}  // namespace onnxruntime
